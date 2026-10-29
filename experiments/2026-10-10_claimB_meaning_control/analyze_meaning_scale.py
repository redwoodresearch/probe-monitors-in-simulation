"""Meaning control (PLAN.md), scale of the shifts: each line's mean run-score shift in no-line run-score SD units, and how
many runs moved toward 'simulated'. Each run is scored by its environment's refit probe, so each shift is divided by that
environment's no-line SD; the SD pooled over all 150 runs (this script's first version) is reported beside it. With the
planted check's result, the planted shift too. Reads result files only; writes <out dir>/meaning_scale_<stamp>.json.
  python3 experiments/2026-10-10_claimB_meaning_control/analyze_meaning_scale.py <out dir> <control json> [<plant json>]
"""
import hashlib
import json
import statistics as st
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

pr = json.loads(Path(sys.argv[2]).read_text())["per_run"]
plant = json.loads(Path(sys.argv[3]).read_text())["per_run"] if len(sys.argv) > 3 else []
out = {}
for probe in ("L63", "A_layer"):
    none = {e: [r[f"{probe}|none|run"] for r in pr if r["environment"] == e] for e in sorted({r["environment"] for r in pr})}
    sd, pooled = {e: st.pstdev(v) for e, v in none.items()}, st.pstdev([r[f"{probe}|none|run"] for r in pr])
    shifts = {v: [(r[f"{probe}|{v}|run"] - r[f"{probe}|none|run"], r["environment"]) for r in pr]
              for v in ("sim", "prod", "neutral")}
    if plant:
        shifts["planted"] = [(r[f"{probe}|run"], r["environment"]) for r in plant]
    out[probe] = {"sd_within_env": sd, "sd_pooled": pooled, "shifts": {k: {
        "runs": len(v), "mean": st.mean(x for x, _ in v), "within_env_sd_units": st.mean(x / sd[e] for x, e in v),
        "pooled_sd_units": st.mean(x for x, _ in v) / pooled, "toward_simulated": sum(x > 0 for x, _ in v)}
        for k, v in shifts.items()}}
    print(probe, json.dumps(out[probe], indent=1))
out |= {"inputs_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[2:]},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()}
dst = Path(sys.argv[1]) / f"meaning_scale_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(out, indent=1))
print("->", dst)
