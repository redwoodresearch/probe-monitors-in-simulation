"""Supersession map records for one paired-rerun launch (09 Append 35 B): keys and shas only, one record per arm per listed cell.

ON the host, after the rerun launch has finished (python3, read-only):
  python3 supersession_rows.py <rerun run dir, e.g. ~/phase2/runs/2026-10-15_il_q1_rlr>  > rows.json
Reads <run>.il_meta.json, <run>.rerun_of.json, <run>.original_pairs.json (all written by ilrr_box.sh before the launch)
and <run>/launch.json. Each row: cell; keys run3|<cell> and sim2|<cell>; the ORIGINAL pair (launch_id, run keys, states,
.eval paths and sha256s); the REPLACEMENT pair (this launch's launch_id, run keys, states, .eval paths and sha256s);
rerun_reason; replacement_complete (both replacement runs ended "done"; informational). Per 09 (Append 32 @ 9fd14891a):
for a listed cell BOTH replacement arms are authoritative, with no fallback to the originals even if a replacement fails.
Fails loudly if the rerun has not finished or a rerun cell has no record.
"""

import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1]).expanduser()
meta = json.loads(Path(f"{run}.il_meta.json").read_text())
binds = json.loads(Path(f"{run}.rerun_of.json").read_text())
orig = json.loads(Path(f"{run}.original_pairs.json").read_text())
lj = json.loads((run / "launch.json").read_text())
assert lj.get("finished"), f"{run} has not finished"
runs = lj["runs"]
runs = list(runs.values()) if isinstance(runs, dict) else runs
by_key = {r["key"]: r for r in runs}


def evals(r: dict) -> list[dict]:
    d = Path(r["attempts"][-1]["log_dir"]) if r.get("attempts") else None
    fs = sorted(d.glob("*.eval")) if d and d.is_dir() else []
    return [{"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in fs]


RECORD_FIELDS = ["cell", "arm", "key", "authoritative", "original", "replacement", "replacement_state_done", "rerun_reason",
                 "replacement_cell_list_sha256"]  # one record per arm per listed cell (09, Append 32 @ 9fd14891a)
records = []
for b in binds["rerun_of"]:
    cell = b["cell"]
    o = orig["pairs"][cell]
    for arm, rk, ok in (("run3", f"attack/{cell}", "run3"), ("sim2", f"sim_attack@september/{cell}", "sim2")):
        assert rk in by_key, (cell, arm, "a rerun cell has no record in the rerun launch")
        n = by_key[rk]
        records.append({
            "cell": cell, "arm": arm, "key": f"{arm}|{cell}",
            "authoritative": "replacement",  # 09: both replacement arms authoritative, no fallback to the originals
            "original": {"launch_id": Path(orig["original_launch"]).name, "run_key": o[ok]["run_key"], "state": o[ok]["state"],
                         "evals": o[ok]["evals"], **({"failed_eval_sha256": b["eval_sha256"]} if arm == "sim2" else {})},
            "replacement": {"launch_id": run.name, "run_key": rk, "state": n["state"], "evals": evals(n)},
            "replacement_state_done": n["state"] == "done",
            "rerun_reason": binds["rerun_reason"],
            "replacement_cell_list_sha256": meta["rerun_of_file_sha256"]})
assert len(records) == 2 * len(binds["rerun_of"])
print(json.dumps({"rerun_launch": run.name, "original_launch": Path(orig["original_launch"]).name, "fields": RECORD_FIELDS,
                  "records": records}, indent=1))
