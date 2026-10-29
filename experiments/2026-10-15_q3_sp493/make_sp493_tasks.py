"""S_paper493 real-arm task files (keys only; no approach text): split the 493 cells (plus804 cells[311:]) between two
devboxes, A and B, half and half within each environment by sha256(SALT + cell). Order each box's cells so every
environment is spread evenly over the file (the frozen launcher starts tasks in file order, 90 s apart per
environment). Approach fields move to q2_approach_* so record_fleet_approaches.py stores each text ON THE DEVBOX and
the two hashes are compared, as for the +311 (make_tasks_311.py).

Also writes honest_keys.json: the r1 entries that keep the launch's default real arms [honest_a, attack, honest_b].
That is the first cell (by key) of each main whose honest_a is missing (35 new mains, plus the T196 mains of t115 and
t98: B cells t619 and t699). Every other r1 entry gets real_arms [attack] after add_r2.py (add_sp493_arms.py).

  uv run --no-project python3 make_sp493_tasks.py <plus804 json> <honest coverage json> <out dir> [boxes, default AB]
v2 (d9 17:35): boxes ABC = A scaleup4, B xbox1, C realhonest; the same per-environment round-robin by sha256(SALT + cell).
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

SALT = "q3-sp493-v1:"
src, cov, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
BOXES = sys.argv[4] if len(sys.argv) > 4 else "AB"
assert BOXES in ("AB", "ABC"), BOXES
h = lambda s: hashlib.sha256((SALT + s).encode()).hexdigest()  # noqa: E731
cells = json.loads(src.read_text())["cells"]
assert len(cells) == 804
b493 = cells[311:]
assert len(b493) == 493 and len({c["cell"] for c in b493}) == 493

coverage = json.loads(cov.read_text())
need_a = {(m["env"], m["main"]) for m in coverage["mains"] if m["honest_a"]["sp493_action"] == "generate"}
need_b = {(m["env"], m["main"]) for m in coverage["mains"] if m["honest_b"]["sp493_action"] == "generate"}
assert need_b <= need_a, "a main needs honest_b but not honest_a; the per-entry arms can't express that"
honest_keys = []
for m in sorted(need_a):
    ks = sorted(c["key"] for c in b493 if (c["env"], c["main"]) == m)
    honest_keys.append(ks[0])

boxes = {b: [] for b in BOXES}
for env in sorted({c["env"] for c in b493}):
    for i, c in enumerate(sorted((c for c in b493 if c["env"] == env), key=lambda c: h(c["cell"]))):
        boxes[BOXES[i % len(BOXES)]].append(c)
summary = {"source": {"file": src.name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "slice": "cells[311:]"},
           "coverage": {"file": cov.name, "sha256": hashlib.sha256(cov.read_bytes()).hexdigest()},
           "salt": SALT, "box_letters": BOXES, "boxes": {}}
for b, cs in boxes.items():
    n_env = Counter(c["env"] for c in cs)
    rank = Counter()
    keyed = []
    for c in sorted(cs, key=lambda c: h(c["cell"])):
        keyed.append(((rank[c["env"]] + 0.5) / n_env[c["env"]], h(c["env"]), c))
        rank[c["env"]] += 1
    ordered = []
    for _, _, c in sorted(keyed, key=lambda x: (x[0], x[1])):
        c = dict(c)
        c["q2_approach_sha256"], c["q2_approach_chars"] = c.pop("approach_sha256"), c.pop("approach_chars")
        c.pop("approach_s3")
        ordered.append(c)
    f = out / f"tasks_sp493_box{b}.json"
    assert not f.exists(), f"{f} exists"
    f.write_text(json.dumps({"rule": __doc__.split("\n\n")[0], "salt": SALT, "box": b,
                             "source": summary["source"], "cells": ordered}, indent=1))
    hk = [c["key"] for c in ordered if c["key"] in honest_keys]
    summary["boxes"][b] = {"file": f.name, "cells": len(ordered), "honest_cells": len(hk),
                           "empty_approach": sum(int(c["q2_approach_chars"]) == 0 for c in ordered),
                           "by_env": dict(sorted(n_env.items())), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
hf = out / "honest_keys_sp493.json"
assert not hf.exists()
hf.write_text(json.dumps({"rule": "r1 entries that keep the default real arms [honest_a, attack, honest_b]; all other r1 "
                                  "entries get real_arms [attack] (add_sp493_arms.py)",
                          "coverage_sha256": summary["coverage"]["sha256"], "keys": sorted(honest_keys)}, indent=1))
summary["honest_keys"] = {"file": hf.name, "n": len(honest_keys), "sha256": hashlib.sha256(hf.read_bytes()).hexdigest()}
assert sum(v["cells"] for v in summary["boxes"].values()) == 493
real = {"attack_r1": 493, "attack_r2": 493, "honest_a": len(honest_keys), "honest_b": len(honest_keys)}
summary["real_runs"] = dict(real, total=sum(real.values()))
(out / "tasks_sp493_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
