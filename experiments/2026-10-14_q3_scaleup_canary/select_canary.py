"""Draw the scale-up canary's 20 cells from q2's +804 list (RULE.md): pool = cells whose main task is not one of the
196's; per environment, in sha256("q3-canary-v1:" + cell) order, the first strategy-bearing cell and the first
empty-approach cell on another main (one of each per environment; an environment without a cell of one kind gives two of
the other, on two mains). Keys, envs and counts only; the approach
fields are moved to q2_approach_* so record_fleet_approaches.py stores each text and the two hashes can be compared.

  python select_canary.py <q2 +804 json> <out json>
"""

import hashlib
import json
import sys
from pathlib import Path

SALT = "q3-canary-v1:"
src, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), f"{out} exists"
data = json.loads(src.read_text())
pool = [c for c in data["cells"] if not c["flags"]["main_in_196"]]
order = sorted(pool, key=lambda c: hashlib.sha256((SALT + c["cell"]).encode()).hexdigest())
chosen = []
for env in sorted({c["env"] for c in pool}):
    mains = set()
    for want_empty in (False, True):
        kinds = (want_empty, not want_empty)  # the wanted kind, else the other one
        c = next(c for k in kinds for c in order
                 if c["env"] == env and (c["approach_chars"] == 0) == k and c["main"] not in mains)
        mains.add(c["main"])
        chosen.append(c)
cells = []
for c in sorted(chosen, key=lambda c: hashlib.sha256((SALT + c["cell"]).encode()).hexdigest()):  # launch order
    c = dict(c)
    c["q2_approach_sha256"], c["q2_approach_chars"] = c.pop("approach_sha256"), c.pop("approach_chars")
    c.pop("approach_s3")
    cells.append(c)
assert len(cells) == 20 and len({c["main"] for c in cells}) == 20
print(f"{len(pool)} pool cells; chosen {len(cells)}: "
      f"{sum(c['q2_approach_chars'] > 0 for c in cells)} strategy-bearing, {sum(c['q2_approach_chars'] == 0 for c in cells)} empty")
for c in cells:
    print(f"  {c['key']} {c['env']} approach_chars {c['q2_approach_chars']}")
out.write_text(json.dumps({"rule": __doc__.split("\n\n")[0], "salt": SALT,
                           "source": {"file": src.name, "sha256": hashlib.sha256(src.read_bytes()).hexdigest()},
                           "cells": cells}, indent=1))
