"""Q4 with the canary's 8 overlap cells (RULE.md Amendment 5; 0d cancelled the canary, d9 16:00): Q4's 77 cells
(config/tasks_311_q4_built.json) plus the 8 +311 cells the canary had (their entries from the canary's
tasks_canary20_final.json, recorded and template-hashed on the canary box, 20/20 built there), each environment spread
evenly over the file as in make_quarters_311.py. Writes config/tasks_311_q4c_built.json; add_r2.py then makes
config/tasks_311_q4c_launch.json. Keys, hashes and counts only.
  python make_q4_with_canary.py <q4 built> <canary final tasks> <q2 +311 json> <out>"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path
SALT = "q3-scaleup311-v1:"
h = lambda s: hashlib.sha256((SALT + s).encode()).hexdigest()  # noqa: E731
q4, can, plus, out = (Path(x) for x in sys.argv[1:5])
d4 = json.loads(q4.read_text())
cells_plus = {c["cell"] for c in json.loads(plus.read_text())["cells"]}
add = [c for c in json.loads(can.read_text())["cells"] if c["cell"] in cells_plus]
assert len(add) == 8, len(add)
keys = set(d4["cells"][0])
for c in add:
    assert set(c) == keys, sorted(set(c) ^ keys)
    assert (c["approach_sha256"], c["approach_chars"]) == (c["q2_approach_sha256"], c["q2_approach_chars"]), c["key"]
    assert c["template_sha256"] == d4["template_sha256"]["with_approach" if c["approach_chars"] else "without_approach"], c["key"]
mine = d4["cells"] + add
assert len({c["cell"] for c in mine}) == len(mine) == 85
n_env, rank, keyed = Counter(c["env"] for c in mine), Counter(), []
for c in sorted(mine, key=lambda c: h(c["cell"])):
    keyed.append(((rank[c["env"]] + 0.5) / n_env[c["env"]], h(c["env"]), c))
    rank[c["env"]] += 1
d4 = dict(d4, cells=[c for _, _, c in sorted(keyed, key=lambda x: (x[0], x[1]))], canary_cells=[c["key"] for c in add],
          rule=__doc__.split("\n\n")[0])
assert not out.exists(), f"{out} exists"
out.write_text(json.dumps(d4, indent=1))
print(json.dumps({"cells": 85, "canary_cells": [c["key"] for c in add], "empty_approach": sum(c["approach_chars"] == 0 for c in mine),
                  "by_env": dict(sorted(n_env.items())), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))
