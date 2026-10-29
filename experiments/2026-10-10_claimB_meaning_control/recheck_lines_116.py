"""Review check for #519 (thread L-Hx), local and free: the planted check covers the 116 eligible runs, but the line
contrasts it is compared with cover all 150. Recomputes the three line shifts and sim - prod (L63 run score) on the same
116 runs from the meaning control's per-run scores, after first reproducing the published 150-run numbers. Run order
and cells come from the CR recompute's per-run flags (stored files sorted by name; each run's main task is checked);
the 116 are the 150 minus the planted check's dropped cells, checked against its per-run main tasks and environments.
Same bootstrap as the result (2,000 main-task draws, seed from "meaning-control-2026-10-10").

  python3 experiments/2026-10-10_claimB_meaning_control/recheck_lines_116.py <out dir> <meaning_control json> \
      <meaning_plant json> <recheck_cr_recompute json>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

res, plant, cr = (json.loads(Path(p).read_text()) for p in sys.argv[2:5])
order, per_run = sorted(cr["per_run_flags"], key=lambda r: r["file"]), res["per_run"]
assert len(order) == len(per_run) == 150 and len({r["cell"] for r in order}) == 150
assert all(r["cell"].split(":")[0] == p["main_task"] for r, p in zip(order, per_run)), "run order does not match the cells"
dropped = {d["cell"] for d in plant["dropped"]}
assert len(dropped) == 34 and dropped <= {r["cell"] for r in order}
kept = [p for r, p in zip(order, per_run) if r["cell"] not in dropped]
assert [(p["main_task"], p["environment"]) for p in kept] == [(p["main_task"], p["environment"]) for p in plant["per_run"]]


def summary(rows, f):
    rng = np.random.default_rng(int.from_bytes(b"meaning-control-2026-10-10"[:8], "big"))
    tasks = sorted({r["main_task"] for r in rows})
    by = {k: [r for r in rows if r["main_task"] == k] for k in tasks}
    boot = [np.mean([f(r) for k in rng.choice(tasks, len(tasks)) for r in by[k]]) for _ in range(2000)]
    return float(np.mean([f(r) for r in rows])), [float(x) for x in np.percentile(boot, [2.5, 97.5])]


out = {}
for name, rows in (("all 150 (published)", per_run), ("the planted check's 116", kept)):
    sh = {v: (lambda r, v=v: r[f"L63|{v}|run"] - r["L63|none|run"]) for v in ("sim", "prod", "neutral")}
    out[name] = {f"shift_{v}": summary(rows, f) for v, f in sh.items()} | {
        "sim_minus_prod": summary(rows, lambda r: sh["sim"](r) - sh["prod"](r)), "runs": len(rows),
        "main_tasks": len({r["main_task"] for r in rows})}
    print(name, json.dumps(out[name]))
pub = res["results"]["L63|run"]
for k in ("shift_sim", "shift_prod", "shift_neutral", "sim_minus_prod"):  # the published numbers, reproduced exactly
    assert np.isclose(out["all 150 (published)"][k][0], pub[k]["mean"], rtol=0, atol=1e-12), k
    assert np.allclose(out["all 150 (published)"][k][1], pub[k]["ci95"], rtol=0, atol=1e-12), k
bound = out["all 150 (published)"]["sim_minus_prod"][1][1]
assert np.isclose(bound, plant["bound_sim_minus_prod_upper"], rtol=0, atol=1e-12)  # the bound the rule used
p_low = plant["results"]["L63|run"]["ci95"][0]
out["planted_lower_bound"] = p_low
out["sensitive_rule_on_116"] = bool(p_low > 0 and p_low > out["the planted check's 116"]["sim_minus_prod"][1][1])
here = Path(__file__)
out |= {"inputs_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[2:5]},
        "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=here.parent).stdout.strip()}
print({k: out[k] for k in ("planted_lower_bound", "sensitive_rule_on_116")})
stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')
dst = Path(sys.argv[1]) / f"recheck_lines_116_{stamp}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(out, indent=1))
print("->", dst)
