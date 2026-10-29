"""Follow-up (simulated-only) task files (RULE.md Amendment 7): each quarter's launch file with its `<key>_r2` entries
removed (they would run real attacks), every other entry byte-identical. Keys and hashes only.
  python make_fu_tasks.py <quarter launch json> <out json>"""
import hashlib, json, sys
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), f"{out} exists"
d = json.loads(src.read_text())
r1 = [c for c in d["cells"] if "real_arms" not in c]
assert len(r1) * 2 == len(d["cells"]) and all(not c["key"].endswith("_r2") for c in r1)
d = dict(d, cells=r1, followup={"from": src.name, "from_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                                "rule": "the quarter's r1 entries only, unchanged; simulated attacks only (real_arms [])"})
out.write_text(json.dumps(d, indent=1))
print(f"{len(r1)} cells -> {out} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
