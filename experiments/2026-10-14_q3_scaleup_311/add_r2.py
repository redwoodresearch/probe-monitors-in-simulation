"""Add each cell's second real attack as its own `<key>_r2` entry (RULE.md, interleaved second attack): a copy of the
cell's final entry (after record_fleet_approaches.py and add_template_hashes.py) whose only differences are the key and
`real_arms: ["attack"]`, placed LAG entries after its cell so both attacks run in the same launch and window. Every other
field is checked equal (canonical JSON). Also tags each entry `attack_pair_launch` ("same") and its box. Keys and hashes
only; runs on the devbox or the laptop (no approach text is read).

  python add_r2.py <final tasks json> <out json> [--lag 10]
"""

import argparse
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--lag", type=int, default=10)
a = ap.parse_args()
assert not a.out.exists(), f"{a.out} exists"
data = json.loads(a.src.read_text())
cells = data["cells"]
assert len({c["key"] for c in cells}) == len({c["cell"] for c in cells}) == len(cells), "keys or cells repeat"
assert not any("real_arms" in c or c["key"].endswith("_r2") for c in cells), "already has r2 entries"
canon = lambda c: json.dumps({k: v for k, v in c.items() if k not in ("key", "real_arms")}, sort_keys=True)  # noqa: E731
r1 = [dict(c, attack_pair_launch="same") for c in cells]
r2 = [dict(c, key=c["key"] + "_r2", real_arms=["attack"]) for c in r1]
order = []
for i, c in enumerate(r1):
    order.append(c)
    if i >= a.lag:
        order.append(r2[i - a.lag])
order += r2[max(0, len(r1) - a.lag):]
assert len(order) == 2 * len(r1) and len({c["key"] for c in order}) == len(order)
by = {c["key"]: c for c in order}
for c in r1:
    assert canon(by[c["key"] + "_r2"]) == canon(c), c["key"]
    assert by[c["key"] + "_r2"]["real_arms"] == ["attack"] and "real_arms" not in c
data["cells"] = order
data["r2"] = {"lag": a.lag, "rule": "each <key>_r2 = its <key> entry with key and real_arms changed only",
              "source_sha256": hashlib.sha256(a.src.read_bytes()).hexdigest()}
a.out.write_text(json.dumps(data, indent=1))
print(f"{len(r1)} cells + {len(r2)} r2 entries (lag {a.lag}) -> {a.out} "
      f"sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
