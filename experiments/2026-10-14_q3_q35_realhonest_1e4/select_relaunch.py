"""The relaunch's task file (RULE.md, Amendment 9), fixed by infrastructure events alone, before any re-run. On the devbox:
    ~/sims/simulated-control-evals-sept/.venv/bin/python select_relaunch.py <launch 1 launch.json> <980-cell tasks file> \
        <out tasks file> <out selection record>
Reads only launch 1's run states, servers and attempt timestamps (never an outcome, transcript or partial content):
- not run: state "not run" (never started; the gate's Ctrl+C at 15:26:10 left them);
- (a) server 0 down: the run's server is server 0 (ap-HKUwBpb4Y86hMDVEA5dHWH, stopped by its laptop billing watcher at
  15:19:51 PT) and its last attempt was still running at or after 15:19:51 (no end, or end >= 15:19:51), whatever it did;
- (b) cut by the gate's Ctrl+C at 15:26:10: state "stopped".
The relaunch runs (a) and (b) again and the never-started ones, as the 980 file's own cells (same keys), in its order.
Prints and writes counts, keys and sha256s; refuses duplicates or any overlap between never-started and affected runs."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

STOP_SERVER0 = datetime.fromisoformat("2026-10-14T15:19:51-07:00").timestamp()
launch_json, tasks_file, out_tasks, out_rec = map(Path, sys.argv[1:5])
assert not out_tasks.exists() and not out_rec.exists(), "outputs exist"
d = json.loads(launch_json.read_text())
server0 = d["servers"][0]
not_run, a, b = [], [], []
for r in d["runs"]:
    if r["state"] == "not run":
        assert not r["attempts"], r["task"]
        not_run.append(r["task"])
        continue
    last = r["attempts"][-1] if r["attempts"] else None
    end = datetime.fromisoformat(last["end"]).timestamp() if last and last.get("end") else None
    if r["server"] == server0 and last and (end is None or end >= STOP_SERVER0):
        a.append(r["task"])
    if r["state"] == "stopped":
        b.append(r["task"])
affected = sorted(set(a) | set(b))
assert len(set(not_run)) == len(not_run) and not set(not_run) & set(affected)
want = set(not_run) | set(affected)
cells = json.loads(tasks_file.read_text())
sel = [c for c in cells["cells"] if c["key"] in want]
assert len(sel) == len(want) == len({c["key"] for c in sel}), "duplicate or missing keys"
raw = tasks_file.read_bytes()
out = json.dumps(cells | {"cells": sel, "relaunch_of": {"launch": str(launch_json.parent.name), "tasks_sha256": hashlib.sha256(raw).hexdigest()}}, indent=1)
out_tasks.write_text(out)
rec = {"launch_json_sha256": hashlib.sha256(launch_json.read_bytes()).hexdigest(), "server0": server0, "stop_server0": "2026-10-14T15:19:51-07:00",
       "counts": {"not_run": len(not_run), "a_server0_down": len(a), "b_cut_by_ctrl_c": len(b), "a_and_b": len(set(a) & set(b)),
                  "affected": len(affected), "relaunch_cells": len(sel)},
       "per_replicate": {f"r{i}": sum(c["key"].endswith(f"_r{i}") for c in sel) for i in range(1, 6)},
       "a_server0_down": sorted(a), "b_cut_by_ctrl_c": sorted(b), "affected": affected,
       "out_tasks_sha256": hashlib.sha256(out.encode()).hexdigest()}
out_rec.write_text(json.dumps(rec, indent=1))
print(json.dumps({k: rec[k] for k in ("counts", "per_replicate", "out_tasks_sha256")}))
