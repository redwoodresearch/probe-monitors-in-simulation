"""The +311 epoch-2 tail split (RULE.md amendment 3; d9/0d GO ~14:00 PT 10-15, before any epoch-2 outcome is read): reads the
launch-1 tasks file ON q3-scaleup4 (~/phase2/ep2_311/pilot_tasks_ep2_311_309.json, sha256 3b645674...), keys and positions only,
and prints the split record: P as a 0-based index into its "cells" list, the keys at index >= P (they belong to ep2_311b), and
the keys below P (launch 1's admitted set).
    ~/sims/simulated-control-evals-sept/.venv/bin/python split_311.py <tasks file> <P>"""
import hashlib
import json
import sys
from pathlib import Path

p, P = Path(sys.argv[1]), int(sys.argv[2])
raw = p.read_bytes()
sha = hashlib.sha256(raw).hexdigest()
assert sha.startswith("3b645674937459b6"), sha
keys = [c["key"] for c in json.loads(raw)["cells"]]
assert len(keys) == 309 == len(set(keys)) and 0 < P < 309
print(json.dumps({"tasks_file": str(p), "tasks_file_sha256": sha, "n_tasks": len(keys), "P_index0": P,
                  "P_positions1": f"{P + 1}-{len(keys)}", "launch2_keys": keys[P:], "launch1_keys": keys[:P],
                  "n_launch2_tasks": len(keys) - P, "n_launch2_runs": 2 * (len(keys) - P)}, indent=1))
