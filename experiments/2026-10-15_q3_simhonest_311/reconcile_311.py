"""Reconcile the +311 simulated-honest cells (the four follow-up task files) against 09's +311 list (il_keylists): per
cell its quarter, main, environment and strategy flag (approach_chars > 0), and every +311 cell missing from the task
files with the reason. Read-only over git; run from the repo root:
python3 experiments/2026-10-15_q3_simhonest_311/reconcile_311.py  (writes results/reconcile_311_cells.json)
"""

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
SCALEUP = "origin/aswin/q3-scaleup-311"
KEYLISTS = "2e75fa8c50978be23f9dc89ee7b04226ef0b4d73:experiments/2026-10-14_il_launch_09/results/il_keylists_2026-10-14.json"
# usable honest_a = the follow-up's simulated run ran (scale-up fork, each follow-up's launch.json, 23:12 PT)
# dropped at the +311 build (origin/aswin/q3-scaleup-311 experiments/2026-10-14_q3_scaleup_311/RULE.md:23)
MISSING_REASON = {"t278_model_registry": "dropped at the +311 build: \"the registry's approach differs from September's\" "
                                         "(scale-up RULE.md:23); in no main or follow-up task file"}
NO_REFERENCE = {"t404_slack_clone": "both real honest runs in the Q3 main errored (EmptyPolicyCompletionError); no pinned reference"}


def show(spec: str) -> str:
    return subprocess.check_output(["git", "show", spec], text=True)


commit = subprocess.check_output(["git", "rev-parse", f"{SCALEUP}^{{commit}}"], text=True).strip()
plus311 = json.loads(show(KEYLISTS))["estimation_set"]["plus311_cells"]
cells = {}
for k in "1234":
    for c in json.loads(show(f"{commit}:experiments/2026-10-14_q3_scaleup_311/config/tasks_311_q{k}_fu.json"))["cells"]:
        assert c["key"] not in cells, c["key"]
        cells[c["key"]] = {"quarter": f"q{k}", "cell": c["cell"], "main": c.get("main"), "env": c["env"],
                           "strategy": bool(c.get("approach_chars")),
                           "usable_honest_a": c["key"] not in NO_REFERENCE}
missing = sorted(set(plus311) - set(cells))
extra = sorted(set(cells) - set(plus311))
out = {"scaleup_commit": commit, "keylists": KEYLISTS, "plus311_listed": len(plus311), "scheduled_cells": len(cells),
       "usable_honest_a": sum(v["usable_honest_a"] for v in cells.values()),
       "no_reference": NO_REFERENCE,
       "missing_from_task_files": {m: MISSING_REASON.get(m, "UNEXPLAINED") for m in missing},
       "extra_not_in_plus311": extra, "cells": dict(sorted(cells.items()))}
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "reconcile_311_cells.json").write_text(json.dumps(out, indent=1) + "\n")
print({k: out[k] for k in ("plus311_listed", "scheduled_cells", "usable_honest_a")}, "missing", missing, "extra", extra)
