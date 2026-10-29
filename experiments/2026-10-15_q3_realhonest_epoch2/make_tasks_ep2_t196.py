"""ON q3-realhonest, in the T196 harness folder (~/phase2/attack_check_rh; no laptop compute): the epoch-2 T196 tasks file. The 194
T196 tasks with a pooled real honest run (config/tasks_ep2_t196_194.json, from 09's FINAL A-or-B pool keys) taken from the 196-task
amendment-9 file in its own order, twice: all with keys <key>_e2a first, then all with <key>_e2b; every other field unchanged.
    ~/sims/simulated-control-evals-sept/.venv/bin/python make_tasks_ep2_t196.py config/pilot_tasks_amendment9_all196.json \
        config/tasks_ep2_t196_194.json config/pilot_tasks_ep2_t196_388.json
Prints both inputs' and the output's sha256."""
import hashlib
import json
import sys
from pathlib import Path

src, sel_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
assert not out.exists(), f"{out} exists"
raw, sel_raw = src.read_bytes(), sel_p.read_bytes()
data, sel = json.loads(raw), set(json.loads(sel_raw)["tasks"])
keys = [c["key"] for c in data["cells"]]
assert len(keys) == 196 == len(set(keys)) and len(sel) == 194 and sel <= set(keys), (len(keys), len(sel), sorted(sel - set(keys)))
base = [c for c in data["cells"] if c["key"] in sel]
cells = [c | {"key": f"{c['key']}_{b}"} for b in ("e2a", "e2b") for c in base]
assert len(cells) == 388 == len({c["key"] for c in cells})
out.write_text(json.dumps(data | {"cells": cells, "epoch": 2, "batches": ["e2a", "e2b"],
                                  "selected_from": {"path": str(src), "sha256": hashlib.sha256(raw).hexdigest()},
                                  "selection": {"path": str(sel_p), "sha256": hashlib.sha256(sel_raw).hexdigest(), "n": 194}}, indent=1))
print(f"{src} {hashlib.sha256(raw).hexdigest()}\n{sel_p} {hashlib.sha256(sel_raw).hexdigest()}\n"
      f"{out} {hashlib.sha256(out.read_bytes()).hexdigest()} (388 cells: 194 x e2a, e2b)")
