"""List every reported point estimate that falls outside its own 95% interval in the matched-false-alarm result files
(the 1% smell: each 1% threshold is a sample maximum, so the bootstrap is biased there). $0, local; reads the files with json.load.

  python3 count_outside_ci.py <result json> [<result json> ...]
"""
import json
import sys
from pathlib import Path


def walk(x, path):
    if isinstance(x, dict):
        if "value" in x and "ci95" in x and isinstance(x["value"], (int, float)):
            yield path, x["value"], x["ci95"]
        for k, v in x.items():
            yield from walk(v, path + [k])


out = {}
for f in sys.argv[1:]:
    with open(f) as fh:
        res = json.load(fh)["results"]
    for rate, r in res.items():
        rows = [(" / ".join(p), v, ci) for p, v, ci in walk(r, []) if not ci[0] <= v <= ci[1]]
        n = sum(1 for _ in walk(r, []))
        out.setdefault(Path(f).name, {})[rate] = {"checked": n, "outside": [{"key": k, "point": v, "ci95": ci} for k, v, ci in rows]}
print(json.dumps(out, indent=1))
