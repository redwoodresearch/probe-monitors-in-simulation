"""Split q2's +311 list into the two scale-up boxes' task files (RULE.md). Drops the canary's cells (their runs count),
splits each environment's cells half and half between the boxes by sha256(SALT + cell), and orders each box's cells so
every environment is spread evenly over the file (the frozen launcher starts tasks in file order and spaces an
environment's launches 90 s apart). Keys, envs and counts only; approach fields move to q2_approach_* so
record_fleet_approaches.py stores each text on the devbox and the two hashes can be compared (as select_canary.py).

  python make_tasks_311.py <q2 +311 json> <canary final tasks json> <out dir>
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

SALT = "q3-scaleup311-v1:"
src, canary, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
h = lambda s: hashlib.sha256((SALT + s).encode()).hexdigest()  # noqa: E731
data = json.loads(src.read_text())
assert len(data["cells"]) == 311 and len({c["main"] for c in data["cells"]}) == 311
canary_cells = {c["cell"] for c in json.loads(canary.read_text())["cells"]}
overlap = sorted(c["key"] for c in data["cells"] if c["cell"] in canary_cells)
pool = [c for c in data["cells"] if c["cell"] not in canary_cells]
assert not any(c["flags"]["main_in_196"] for c in pool), "a +311 cell is on one of the 196's mains"
boxes = {1: [], 2: []}
for env in sorted({c["env"] for c in pool}):
    for i, c in enumerate(sorted((c for c in pool if c["env"] == env), key=lambda c: h(c["cell"]))):
        boxes[1 + i % 2].append(c)
summary = {"source": {"file": src.name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest()},
           "canary": {"file": canary.name, "sha256": hashlib.sha256(canary.read_bytes()).hexdigest()},
           "canary_overlap_keys": overlap, "salt": SALT, "boxes": {}}
for b, cells in boxes.items():
    n_env = Counter(c["env"] for c in cells)
    rank = Counter()
    keyed = []
    for c in sorted(cells, key=lambda c: h(c["cell"])):  # i-th cell of its environment sits at (i + 0.5) / n_env
        keyed.append(((rank[c["env"]] + 0.5) / n_env[c["env"]], h(c["env"]), c))
        rank[c["env"]] += 1
    ordered = []
    for _, _, c in sorted(keyed, key=lambda x: (x[0], x[1])):
        c = dict(c)
        c["q2_approach_sha256"], c["q2_approach_chars"] = c.pop("approach_sha256"), c.pop("approach_chars")
        c.pop("approach_s3")
        ordered.append(c)
    f = out / f"tasks_311_box{b}.json"
    assert not f.exists(), f"{f} exists"
    f.write_text(json.dumps({"rule": __doc__.split("\n\n")[0], "salt": SALT, "box": b,
                             "source": summary["source"], "cells": ordered}, indent=1))
    summary["boxes"][b] = {"file": f.name, "cells": len(ordered), "empty_approach": sum(c["q2_approach_chars"] == 0 for c in ordered),
                           "by_env": dict(sorted(n_env.items())), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
assert sum(v["cells"] for v in summary["boxes"].values()) == len(pool) == 311 - len(overlap)
print(json.dumps(summary, indent=1))
