"""No-pinned-reference cells, read ON an IL host (read-only; structure only). For each key given: every non-real run of
that key in the IL launch.json (run key, state, why, number of attempts), the launch's reference_state entry for the key,
and whether the key is present in the references of the launch's references_from launch.json (the main or the index).
Expected for a cell without a pinned reference (run_pilot.py:668-672 @ bbf07299b): state "not run", why "no pinned
reference in <references_from>", 0 attempts, reference_state "none in the prior run", absent from the main's references.
  python3 no_reference_cells_read.py <IL run dir> <key> ...
"""

import hashlib
import json
import sys
from pathlib import Path

run, keys = Path(sys.argv[1]), sys.argv[2:]
d = json.loads((run / "launch.json").read_text())
runs = d["runs"]
runs = list(runs.values()) if isinstance(runs, dict) else runs
rf = Path(d["references_from"])
main = json.loads((rf / "launch.json").read_text())
rows = []
for k in keys:
    sims = [r for r in runs if not r["real"] and r["key"].split("/", 1)[1] == k]
    rs = (d.get("reference_state") or {}).get(k)
    rows.append({"key": k, "non_real_runs": [{"run_key": r["key"], "state": r["state"], "why": r.get("why"),
                                              "attempts": len(r.get("attempts") or [])} for r in sims],
                 "reference_state": rs.get("state") if isinstance(rs, dict) else rs,
                 "in_main_references": k in (main.get("references") or {}),
                 "expected": bool(sims) and all(r["state"] == "not run" and r.get("why") == f"no pinned reference in {rf}"
                                                and not r.get("attempts") for r in sims)
                             and (rs.get("state") if isinstance(rs, dict) else rs) == "none in the prior run"
                             and k not in (main.get("references") or {})})
print(json.dumps({"launch_run_dir": str(run), "launch_json_sha256_at_read": hashlib.sha256((run / "launch.json").read_bytes()).hexdigest(),
                  "references_from": str(rf), "references_from_launch_json_sha256": hashlib.sha256((rf / "launch.json").read_bytes()).hexdigest(),
                  "rows": rows}))
