"""ON q3-ep2-nm (no laptop compute): the epoch-2 nm360 tasks file, written to ~/phase2/ep2_nm/ (outside the frozen nm harness,
which stays byte-identical to its manifest). make_tasks_ep2_311.py with the nm inputs: the 351 tasks (tasks_ep2_nm_351.json,
select_ep2_nm.py: 360 minus setup_failures v4's 9) taken as their r1 entries (no `real_arms` key) from the five split launch files
S1-S5 in the harness, byte-identical, keys unchanged; each found exactly once. The top-level fields the launcher reads
(`construction`: new_mains_v1, and the others) must be equal across the five files; the split-specific `split` and `r2` are dropped.
    cd ~/phase2/nm/attack_check_nm && ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/ep2_nm/make_tasks_ep2_nm.py \
        ~/phase2/ep2_nm/tasks_ep2_nm_351.json ~/phase2/ep2_nm/pilot_tasks_ep2_nm_351.json config/tasks_nm_s1_launch.json \
        config/tasks_nm_s2_launch.json config/tasks_nm_s3_launch.json config/tasks_nm_s4_launch.json config/tasks_nm_s5_launch.json
Prints every input's and the output's sha256."""
import hashlib
import json
import sys
from pathlib import Path

sel_p, out, srcs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
assert not out.exists(), f"{out} exists"
sel = json.loads(sel_p.read_bytes())["tasks"]
assert len(sel) == 351 == len(set(sel))
found, first = {}, None
for p in srcs:
    d = json.loads(p.read_bytes())
    first = first or d
    for c in d["cells"]:
        if "real_arms" not in c and c["key"] in sel:
            assert c["key"] not in found, f"{c['key']} in two splits"
            found[c["key"]] = c
missing = sorted(set(sel) - set(found))
assert not missing, f"not found: {missing}"
cells = [found[k] for k in sel]
DROP = ("cells", "split", "r2")
metas = [{k: v for k, v in json.loads(p.read_bytes()).items() if k not in DROP} for p in srcs]
assert all(m == metas[0] for m in metas) and metas[0].get("construction") == "new_mains_v1", "split files' metadata differ"
meta = metas[0]
out.write_text(json.dumps(meta | {"cells": cells, "epoch": 2, "selected_from": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in srcs],
                                  "selection": {"path": str(sel_p), "sha256": hashlib.sha256(sel_p.read_bytes()).hexdigest(), "n": 351}}, indent=1))
for p in [sel_p, *srcs]:
    print(p, hashlib.sha256(p.read_bytes()).hexdigest())
print(out, hashlib.sha256(out.read_bytes()).hexdigest(), f"({len(cells)} cells)")
