"""The four-quarter queue (RULE.md, Quarters; d9 and 0d 15:10): the 302 cells that passed the boxes' preflights (the union
of config/tasks_311_box{1,2}_built.json, each recorded and template-hashed on its box) split into quarters Q1-Q4 by
int(sha256(SALT + cell), 16) % 4, the same salt as the box split. Within a quarter each environment's cells are spread
evenly over the file (as make_tasks_311.py). Writes config/tasks_311_q<N>_built.json (cells only); add_r2.py then makes
each quarter's launch file. Keys, hashes and counts only.

  python make_quarters_311.py <box1 built json> <box2 built json> <out dir>
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

SALT = "q3-scaleup311-v1:"
h = lambda s: hashlib.sha256((SALT + s).encode()).hexdigest()  # noqa: E731
b1, b2, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
d1, d2 = json.loads(b1.read_text()), json.loads(b2.read_text())
cells = d1["cells"] + d2["cells"]
assert len(cells) == len({c["cell"] for c in cells}) == len({c["key"] for c in cells}) == 302, len(cells)
summary = {"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (b1, b2)}, "salt": SALT, "quarters": {}}
for q in range(1, 5):
    mine = [c for c in cells if int(h(c["cell"]), 16) % 4 == q - 1]
    n_env, rank, keyed = Counter(c["env"] for c in mine), Counter(), []
    for c in sorted(mine, key=lambda c: h(c["cell"])):
        keyed.append(((rank[c["env"]] + 0.5) / n_env[c["env"]], h(c["env"]), c))
        rank[c["env"]] += 1
    ordered = [c for _, _, c in sorted(keyed, key=lambda x: (x[0], x[1]))]
    f = out / f"tasks_311_q{q}_built.json"
    assert not f.exists(), f"{f} exists"
    f.write_text(json.dumps({"rule": __doc__.split("\n\n")[0], "salt": SALT, "quarter": q,
                             "template_sha256": d1["template_sha256"], "cells": ordered}, indent=1))
    summary["quarters"][q] = {"file": f.name, "cells": len(ordered), "empty_approach": sum(c["approach_chars"] == 0 for c in ordered),
                              "by_env": dict(sorted(n_env.items())), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
assert d1["template_sha256"] == d2["template_sha256"]
assert sum(v["cells"] for v in summary["quarters"].values()) == 302
print(json.dumps(summary, indent=1))
