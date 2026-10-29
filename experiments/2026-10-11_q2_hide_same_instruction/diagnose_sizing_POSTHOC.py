"""Row 11 pilot, post hoc (after pilot_sizing_2026-10-11_224031.json read "too few even at k = 24"): is the plan's
30-run minimum or the noise what stops the full stage? From the pilot's per-run usable rates (the same rates the power
simulation draws from), the exact chance that a full-stage run has both groups at k samples, the expected count of
176 runs with both, and the chance of reaching 30. Reads the readout's labels and usable flags only; no score. Local, $0.

  python3 experiments/2026-10-11_q2_hide_same_instruction/diagnose_sizing_POSTHOC.py <readout json> <pilot sizing json>
"""
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from math import comb
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
N_FULL, MIN_BOTH, K0 = 176, 30, 8


def main(readout_p: Path, sizing_p: Path):
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    ro, sz = json.loads(readout_p.read_text()), json.loads(sizing_p.read_text())
    if sz["readout"] != readout_p.name:
        raise SystemExit("the sizing was made from another readout")
    if sz["scored usable samples out of memory"]:  # the power simulation drops those; these rates would not
        raise SystemExit("usable samples ran out of memory: rates would differ from the power simulation's")
    n = Counter((r["key"], r["label"]) for r in ro["rows"] if r["usable"]); runs = sorted({r["key"] for r in ro["rows"]})
    rates = [(n[(k, "hid")] / K0, n[(k, "named")] / K0) for k in runs]
    out = {"plan": "PLAN.md (this folder); post hoc diagnostic, no verdict", "git": head, "readout": readout_p.name, "sizing": sizing_p.name,
           "pilot runs": len(runs), "full-stage runs": N_FULL, "minimum runs with both groups": MIN_BOTH, "by k": {}}
    for k in (8, 12, 16, 24, 48):
        p = [1 - (1 - h) ** k - (1 - m) ** k + max(0.0, 1 - h - m) ** k for h, m in rates]
        pb = sum(p) / len(p)  # a full-stage run draws a pilot run's rates uniformly, as in the power simulation
        tail = sum(comb(N_FULL, j) * pb ** j * (1 - pb) ** (N_FULL - j) for j in range(MIN_BOTH, N_FULL + 1))
        out["by k"][k] = {"P(a run has both groups)": round(pb, 4), "expected runs with both": round(N_FULL * pb, 1), "P(at least 30)": round(tail, 4)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"sizing_diagnostic_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
