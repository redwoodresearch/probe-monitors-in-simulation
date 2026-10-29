"""Admission check before launch (0d 03:52): the frozen launcher reserves ~$75 (worst_per_sim_run_usd) per unfinished
simulated run against the spend cap and starts tasks in strict file order, so a launch runs about cap / $75 cells at once at
the start and (cap - spend) / $75 later; a cap below $75 admits nothing (il_s2_rlr, cap $41: 0 runs, 03:21:28).
Each registered config must have cap >= $75 x its minimum parallel cells (completions 10; rlr2 and rlr3 min(cells, 5),
addendum 04:08: q4_rlr2 and every config/pilot_ilrr3_<h>.yaml as it is generated); a config that admits
fewer FAILS and the script exits 1. The two original s1/s2 rerun configs are checked as negative controls: they must FAIL.
  python3 check_admission.py > results/il_admission_check_2026-10-15.txt   (exit 0 = every registered config passes)
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
C = HERE / "config"
RESERVE = 75.0
REGISTERED = [(C / "completion" / f"pilot_ilc_{h}.yaml", 10) for h in ("q1", "q2", "q3", "q4", "old")] + \
             [(C / f"pilot_ilrr2_{h}.yaml", 5) for h in ("s1", "s2", "q4")] + \
             [(f, 5) for f in sorted(C.glob("pilot_ilrr3_*.yaml"))]  # min(cells, 5) applied in check()
NEGATIVE = [(C / f"pilot_ilrr_{h}.yaml", 1) for h in ("s1", "s2")]


def check(cfg: Path, min_par: int) -> dict:
    text = cfg.read_text()
    cap = float(re.search(r"(?m)^  cap_usd: ([0-9.]+)", text).group(1))
    limit = float(re.search(r"(?m)^  worst_case_limit_usd: ([0-9.]+)", text).group(1))
    tasks = re.search(r"(?m)^tasks: (\S+)", text).group(1)
    cells = len(json.loads((HERE / tasks.replace("config/", "config/completion/", 1) if "tasks_ilc_" in tasks else HERE / tasks).read_text())["cells"])
    need = RESERVE * min(min_par, cells)
    return {"config": str(cfg.relative_to(HERE)), "cells": cells, "cap_usd": cap, "worst_case_limit_usd": limit,
            "min_parallel": min(min_par, cells), "needed_usd": need, "admitted_at_start": min(cells, int(cap // RESERVE)),
            "pass": cap >= need and limit == cap}


rows = [check(c, k) | {"kind": "registered"} for c, k in REGISTERED] + [check(c, k) | {"kind": "negative control"} for c, k in NEGATIVE]
for r in rows:
    print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['kind']:16s} {r['config']:45s} cells {r['cells']:3d}  cap ${r['cap_usd']:7.0f}"
          f"  needs >= ${r['needed_usd']:5.0f} ({r['min_parallel']} x ${RESERVE:.0f})  admits {r['admitted_at_start']} at start")
ok = all(r["pass"] for r in rows if r["kind"] == "registered") and not any(r["pass"] for r in rows if r["kind"] == "negative control")
print(json.dumps({"all_registered_pass": all(r["pass"] for r in rows if r["kind"] == "registered"),
                  "negative_controls_fail": not any(r["pass"] for r in rows if r["kind"] == "negative control"), "rows": rows}))
sys.exit(0 if ok else 1)
