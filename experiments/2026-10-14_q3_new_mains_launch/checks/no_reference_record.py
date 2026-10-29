"""ON THE BOX, read-only (q2's follow-up record, 2026-10-15): for a main launch and its follow-up, every simulated run
recorded "not run" for want of an honest_a reference. Per cell: the simulated run key in both launches, its state, the
launcher's why string, attempts, the main's reference_state entry for the task (tries, state, unusable run folders), and
the real arms' states. Keys, paths and hashes only. Prints one JSON object.
  python3 no_reference_record.py <main run dir> <follow-up run dir>"""

import hashlib
import json
import sys
from pathlib import Path

main, fu = Path(sys.argv[1]), Path(sys.argv[2])
rec = {}
launches = {}
for tag, run in (("main", main), ("follow_up", fu)):
    raw = (run / "launch.json").read_bytes()
    d = json.loads(raw)
    launches[tag] = d
    rec[f"{tag}_launch_id"] = run.name
    rec[f"{tag}_launch_json_sha256"] = hashlib.sha256(raw).hexdigest()
cells = sorted({r["task"] for tag in launches for r in launches[tag]["runs"]
                if not r["real"] and r["state"] == "not run" and "reference" in r.get("why", "")})
rec["cells"] = []
for t in cells:
    row = {"task": t, "reference_state_main": launches["main"]["reference_state"].get(t)}
    for tag, d in launches.items():
        row[tag] = [{"key": r["key"], "real": r["real"], "state": r["state"], "why": r.get("why"), "attempts": len(r["attempts"])}
                    for r in d["runs"] if r["task"] == t]
    rec["cells"].append(row)
rec["n_cells"] = len(cells)
print(json.dumps(rec, indent=1))
