"""Read-only key check (09 via d9 10-15, option (b) and (a)): writes a box script that, on the main launch's OWN box, re-hashes every
counted honest_a/honest_b .eval a pinned main manifest lists and reports, per key, whether its sha256 equals the manifest's
(counts and mismatching keys only). Keys and hashes only; never opens a transcript.

  python3 probe_main_donors.py <manifest commit> <manifest path> > probe.sh   # then: ssm_run_i.sh <main box> "$(cat probe.sh)"
"""
import json
import subprocess
import sys

commit, path = sys.argv[1], sys.argv[2]
man = json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout)
rows = [[f"{f['arm_dir']}/{f['task']}", f["path"], f["sha256"]] for f in man["files"] if f["arm_dir"] in ("honest_a", "honest_b")]
roots = sorted({f["launch"] for f in man["files"]})
print(f"""date '+probe at %Y-%m-%d %H:%M:%S %Z'; hostname
python3 - <<'PY'
import hashlib, json, os
rows = {json.dumps(rows)}
base = [os.path.expanduser(p) for p in ("~/phase2/runs", "~/phase2/nm/runs")]
out = {{"rows": len(rows), "match": 0, "missing": [], "mismatch": []}}
for key, rel, want in rows:
    p = next((os.path.join(b, rel) for b in base if os.path.exists(os.path.join(b, rel))), None)
    if p is None:
        out["missing"].append(key); continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h == want: out["match"] += 1
    else: out["mismatch"].append(key)
print(json.dumps(out))
PY""")
