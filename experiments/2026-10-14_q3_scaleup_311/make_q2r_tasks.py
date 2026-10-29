"""The Q2 rerun's task file (RULE.md Amendment 11): the cells of Q2's first follow-up launch (2026-10-14_173012_scaleup311_q2_fu)
whose simulated run is "stopped" (cut by gate1_311b.py's Ctrl+C at 19:30:33 PT) or "not run" in its FINAL launch.json,
selected by state from results/q2_fu_final_states_2026-10-14.json (keys, states, tasks and attempt counts only, read on box
2 from that launch.json, sha256 6359c273...). Every selected cell is byte-identical to config/tasks_311_q2_fu.json's.
The 20 "done" cells are not rerun. Keys and hashes only.
  python make_q2r_tasks.py <tasks_311_q2_fu.json> <final states json> <out json>"""
import hashlib, json, sys
from pathlib import Path
src, states_f, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
assert not out.exists(), f"{out} exists"
d, states = json.loads(src.read_text()), json.loads(states_f.read_text())
by_task = {v["task"]: v["state"] for v in states.values()}
assert len(by_task) == len(states) == len(d["cells"]) == 75 and set(by_task) == {c["key"] for c in d["cells"]}
assert set(by_task.values()) == {"done", "stopped", "not run"}, set(by_task.values())
cells = [c for c in d["cells"] if by_task[c["key"]] in ("stopped", "not run")]
n = {s: sum(v == s for v in by_task.values()) for s in ("done", "stopped", "not run")}
assert len(cells) == n["stopped"] + n["not run"] == 55 and n["done"] == 20, n
d = dict(d, cells=cells, rerun={
    "of_launch": "2026-10-14_173012_scaleup311_q2_fu", "of_launch_json_sha256": "6359c27350c0335ead56ea51b423ae6e9c86731e7822582edce376423ba533ba",
    "from": src.name, "from_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    "states_sha256": hashlib.sha256(states_f.read_bytes()).hexdigest(), "counts_in_first_launch": n,
    "rule": "cells whose simulated run is 'stopped' (cut by the usable gate's Ctrl+C) or 'not run' in the first launch's "
            "final launch.json; selected on in-flight status, which correlates with run duration and possibly outcome; "
            "the 20 'done' cells (2 unusable among them) are kept and not rerun",
    "stopped": sorted(c["key"] for c in cells if by_task[c["key"]] == "stopped"),
    "not_run": sorted(c["key"] for c in cells if by_task[c["key"]] == "not run")})
out.write_text(json.dumps(d, indent=1))
print(f"{len(cells)} cells ({n['stopped']} stopped + {n['not run']} not run) -> {out} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
