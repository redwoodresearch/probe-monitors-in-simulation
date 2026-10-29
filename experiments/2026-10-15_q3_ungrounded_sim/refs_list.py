"""On a box: print the pinned reference copies of the wanted task keys from a launch.json (relay_refs_ug.sh).
  python3 - <launch.json> <comma-separated keys> < refs_list.py"""
import json
import re
import sys

refs = json.load(open(sys.argv[1]))["references"]
want = set(sys.argv[2].split(","))
paths = sorted(r["copy"] for k, r in refs.items() if k in want)
assert all(re.fullmatch(r"/[A-Za-z0-9_./-]+\.eval", p) for p in paths), paths  # plain paths only: safe as tar arguments
print("\n".join(paths))
