"""Writes the 980-cell tasks file of the one-launch design (RULE.md, Amendment 2): the 196 amendment-9 cells x 5 in
replicate order (all of r1 first, then r2, ...), keys <key>_r1..<key>_r5, every other field unchanged. Run on the devbox
in the harness folder (no laptop compute):
    ~/sims/simulated-control-evals-sept/.venv/bin/python make_tasks_980.py \
        config/pilot_tasks_amendment9_all196.json config/pilot_tasks_rh1e4_980.json
Prints the input's and the output's sha256."""
import hashlib
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), f"{out} exists"
raw = src.read_bytes()
data = json.loads(raw)
assert len(data["cells"]) == 196 and len({c["key"] for c in data["cells"]}) == 196
cells = [c | {"key": f"{c['key']}_r{i}"} for i in range(1, 6) for c in data["cells"]]
assert len({c["key"] for c in cells}) == 980
out.write_text(json.dumps(data | {"cells": cells, "replicates": 5,
                                  "replicated_from": {"path": str(src), "sha256": hashlib.sha256(raw).hexdigest()}},
                          indent=1))
print(f"{src} {hashlib.sha256(raw).hexdigest()}\n{out} {hashlib.sha256(out.read_bytes()).hexdigest()} (980 cells)")
