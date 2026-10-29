"""Q2 rerun audit manifest for 09 (RULE.md Amendment 11), in 09's schema, by script from: Q2's first follow-up's final
states (results/q2_fu_final_states_2026-10-14.json), its unusable runs (results/unusable_runs_final_fu_q2_*.jsonl,
unusable_runs.py on its final launch.json) and the rerun's task file (config/tasks_311_q2r_fu.json). "cell" is the task
key (e.g. t271_slack_clone), as in every task file; a run key is "<arm>@<simulator>/<cell>", as in launch.json. The
counted fields of the cut and never-started cells name the RERUN's run and launch; they are null until the rerun has
launched (pass its run directory name as the 4th argument). Keys only.
  python make_q2_audit.py <final states json> <unusable jsonl> <rerun task file> [<rerun launch id>] <out json>"""
import hashlib, json, sys
from pathlib import Path
states_f, unusable_f, tasks_f, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[-1])
rerun = sys.argv[4] if len(sys.argv) == 6 else None
FIRST = "2026-10-14_173012_scaleup311_q2_fu"
states, u = json.loads(states_f.read_text()), json.loads(unusable_f.read_text())
assert u["run"] == FIRST and len(states) == 75
tasks = json.loads(tasks_f.read_text())
assert tasks["rerun"]["of_launch"] == FIRST
rerun_cells = {c["key"] for c in tasks["cells"]}
row = lambda k: k.split("/", 1)[1]
counted = lambda cell: {"counted_run_key": f"sim_attack@september/{cell}" if rerun else None, "counted_launch_id": rerun}
ended = [{"cell": v["task"], "run_key": k, "launch_id": FIRST} for k, v in sorted(states.items()) if v["state"] == "done"]
cut = [{"cell": v["task"], "original_run_key": k, "original_launch_id": FIRST, **counted(v["task"])}
       for k, v in sorted(states.items()) if v["state"] == "stopped"]
never = [{"cell": v["task"], **counted(v["task"])} for k, v in sorted(states.items()) if v["state"] == "not run"]
unusable = sorted(row(x["key"]) for x in u["unusable"])
assert all(k == f"sim_attack@september/{v['task']}" for k, v in states.items())
assert {r["cell"] for r in cut} | {r["cell"] for r in never} == rerun_cells and not rerun_cells & {r["cell"] for r in ended}
assert set(unusable) <= {r["cell"] for r in ended}
counts = {"ended": len(ended), "cut": len(cut), "never_started": len(never), "unusable_ended": len(unusable),
          "rerun_cells": len(rerun_cells), "cells": len(states)}
assert counts == {"ended": 20, "cut": 19, "never_started": 36, "unusable_ended": 2, "rerun_cells": 55, "cells": 75}, counts
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
doc = {"ended": ended, "cut": cut, "never_started": never, "unusable_ended": unusable, "counts": counts,
       "sources": {"first_launch": FIRST, "first_launch_json_sha256": tasks["rerun"]["of_launch_json_sha256"],
                   "states": states_f.name, "states_sha256": sha(states_f), "unusable": unusable_f.name,
                   "unusable_sha256": sha(unusable_f), "rerun_tasks": tasks_f.name, "rerun_tasks_sha256": sha(tasks_f),
                   "rerun_launch": rerun},
       "notes": "cut = in flight at the usable gate's Ctrl+C (19:30:33 PT), EXCLUDED records with cause 'gate Ctrl+C'; "
                "the 55 rerun cells were selected on in-flight status, which correlates with run duration and possibly outcome"}
out.write_text(json.dumps(doc, indent=1))
print(f"{counts} -> {out} sha256 {sha(out)}")
