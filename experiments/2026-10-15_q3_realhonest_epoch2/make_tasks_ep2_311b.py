"""ON q3-realhonest (no laptop compute): the ep2_311b tasks file (RULE.md amendment 3), written to ~/phase2/ep2_311b/. Launch 1's
tasks file (rebuilt here by stage_ep2_311b.sh from the same committed inputs; its sha256 must be 3b645674..., the file the split
record names) sliced at P = 265: its cells[265:], byte-identical, same order, and its top-level fields; the keys must equal the
split record's launch2_keys.
    ~/sims/simulated-control-evals-sept/.venv/bin/python make_tasks_ep2_311b.py <launch-1 tasks file> <split record> <out>
Prints every input's and the output's sha256."""
import hashlib
import json
import sys
from pathlib import Path

src, rec_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
assert not out.exists(), f"{out} exists"
raw, rec_raw = src.read_bytes(), rec_p.read_bytes()
sha = hashlib.sha256(raw).hexdigest()
rec = json.loads(rec_raw)
assert sha == rec["tasks_file_sha256"] and sha.startswith("3b645674937459b6"), sha
d = json.loads(raw)
P = rec["P"]["index0"]
assert P == 265
cells = d["cells"][P:]
assert [c["key"] for c in cells] == rec["launch2_keys"] and len(cells) == 44
out.write_text(json.dumps({k: v for k, v in d.items() if k != "cells"} | {
    "cells": cells, "split": {"launch": "2026-10-15_ep2_311b", "P_index0": P, "from": {"path": str(src), "sha256": sha},
                              "record": {"path": str(rec_p), "sha256": hashlib.sha256(rec_raw).hexdigest()}}}, indent=1))
for p in (src, rec_p, out):
    print(p, hashlib.sha256(p.read_bytes()).hexdigest())
print(f"({len(cells)} cells)")
