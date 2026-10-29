"""Review check for #526 (thread LWts), local and free: the meaning control's published summary recomputed without the
runs whose site line went to a different boundary (bare \\r; recheck_cr_sites_modal.py's flags). It first reproduces the
published numbers from the result file's per-run scores (run order = the stored files sorted by name, each name the
cell's sha256 prefix; every run's main task is checked against its cell), then drops the flagged runs. Same bootstrap as
the result (2,000 main-task draws, seed from "meaning-control-2026-10-10").

  python3 experiments/2026-10-10_claimB_meaning_control/recheck_cr_recompute.py <meaning_control result json> <per-run flags json> <out dir>
Writes <out dir>/recheck_cr_recompute_<stamp>.json: both summaries, the per-run flags (cells only) and provenance.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

res, flags = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
order = sorted(flags, key=lambda r: r["file"])
per_run = res["per_run"]
assert len(order) == len(per_run) == 150
assert all(r["cell"].split(":")[0] == p["main_task"] for r, p in zip(order, per_run)), "run order does not match the cells"
assert all(r["stored"] and r.get("matches_stored") for r in flags) and len({r["cell"] for r in flags}) == len({r["file"] for r in flags}) == 150, "flags must be 150 unique, stored, matching sites"
LINES = ("sim", "prod", "neutral")


def summary(rows, f):
    rng = np.random.default_rng(int.from_bytes(b"meaning-control-2026-10-10"[:8], "big"))
    tasks = sorted({r["main_task"] for r in rows})
    by = {k: [r for r in rows if r["main_task"] == k] for k in tasks}
    draws = [[r for k in rng.choice(tasks, len(tasks)) for r in by[k]] for _ in range(2000)]
    boot = [np.mean([f(r) for r in d]) for d in draws]
    return round(float(np.mean([f(r) for r in rows])), 5), [round(float(x), 5) for x in np.percentile(boot, [2.5, 97.5])]


out = {}
for name, rows in (("all 150 (published)", per_run), ("without the flagged runs", [p for r, p in zip(order, per_run) if r["same_boundary"]])):
    sh = {v: (lambda r, v=v: r[f"L63|{v}|run"] - r["L63|none|run"]) for v in LINES}
    block = {f"shift_{v}": summary(rows, f) for v, f in sh.items()}
    block["sim_minus_prod"] = summary(rows, lambda r: sh["sim"](r) - sh["prod"](r))
    block["runs"] = len(rows)
    m = {v: block[f"shift_{v}"][0] for v in LINES}
    block["reads_meaning"] = bool(block["sim_minus_prod"][1][0] > 0 and m["prod"] <= m["neutral"] <= m["sim"])
    if name.startswith("all 150"):  # the published numbers must be reproduced before any run is dropped
        pub = res["results"]["L63|run"]
        for k in ("shift_sim", "shift_prod", "shift_neutral", "sim_minus_prod"):
            assert block[k][0] == round(pub[k]["mean"], 5) and block[k][1] == [round(x, 5) for x in pub[k]["ci95"]], k
    out[name] = block
here = Path(__file__)
res_out = {"summaries_L63_run": out, "flags_from": "recheck_cr_sites_modal.py, app ap-UGgMXpLCm35KplufGkzIUH (stdout, saved verbatim)",
           "per_run_flags": flags, "inputs_sha256": {Path(a).name: hashlib.sha256(Path(a).read_bytes()).hexdigest() for a in sys.argv[1:3]},
           "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=here.parent).stdout.strip()}
path = Path(sys.argv[3]) / f"recheck_cr_recompute_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
with open(path, "x") as fh:
    fh.write(json.dumps(res_out, indent=1))
print(json.dumps(out, indent=1))
print("->", path)
print("published L63|run:", {k: (round(v["mean"], 5), [round(x, 5) for x in v["ci95"]]) for k, v in res["results"]["L63|run"].items() if isinstance(v, dict)})
