"""ON q3-scaleup4 (no laptop compute): the epoch-2 +311 tasks file, written to ~/phase2/ep2_311/ (outside the frozen harness, which
stays byte-identical to its manifest). The 309 +311 tasks with a pooled real honest run (tasks_ep2_311_309.json, from 09's FINAL
A-or-B pool keys) taken as their r1 entries (no `real_arms` key) from the launch files the four quarters actually ran with,
byte-identical, keys unchanged (the launch id marks the epoch): Q1-Q3's from the harness (@ 652db965e), Q4's from the 85-cell file
Q4 ran with (sha256 033d4beb..., Amendment 5: Q4's 77 cells plus the canary's 8; the harness copy holds the original 77-cell file,
which lacks 8 of the 309). Each task must be found exactly once.
    cd ~/phase2/attack_check_311 && ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/ep2_311/make_tasks_ep2_311.py \
        ~/phase2/ep2_311/tasks_ep2_311_309.json ~/phase2/ep2_311/pilot_tasks_ep2_311_309.json config/tasks_311_q1_launch.json \
        config/tasks_311_q2_launch.json config/tasks_311_q3_launch.json ~/phase2/ep2_311/tasks_311_q4_launch_85.json
Prints every input's and the output's sha256."""
import hashlib
import json
import sys
from pathlib import Path

sel_p, out, srcs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
assert not out.exists(), f"{out} exists"
sel = json.loads(sel_p.read_bytes())["tasks"]
assert len(sel) == 309 == len(set(sel))
found, first = {}, None
for p in srcs:
    d = json.loads(p.read_bytes())
    first = first or d
    for c in d["cells"]:
        if "real_arms" not in c and c["key"] in sel:
            assert c["key"] not in found, f"{c['key']} in two quarters"
            found[c["key"]] = c
missing = sorted(set(sel) - set(found))
assert not missing, f"not found: {missing}"
cells = [found[k] for k in sel]
meta = {k: v for k, v in first.items() if k not in ("cells", "quarter", "r2")}
out.write_text(json.dumps(meta | {"cells": cells, "epoch": 2, "selected_from": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in srcs],
                                  "selection": {"path": str(sel_p), "sha256": hashlib.sha256(sel_p.read_bytes()).hexdigest(), "n": 309}}, indent=1))
for p in [sel_p, *srcs]:
    print(p, hashlib.sha256(p.read_bytes()).hexdigest())
print(out, hashlib.sha256(out.read_bytes()).hexdigest(), f"({len(cells)} cells)")
