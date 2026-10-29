"""The 351 nm360 tasks of epoch 2 (RULE.md; 09's Append 38 @ 0f5d1b429: "360 minus v4's 9 setup-failure cells"), keys only, from
committed git objects: the r1 entries (no `real_arms`) of the five split launch files S1-S5 (the reruns 4d/4r/5r re-ran S4/S5
cells with byte-identical entries) minus setup_failures v4's 9 cells. Writes config/tasks_ep2_nm_351.json.
    python3 select_ep2_nm.py   (from this folder, in the repo worktree)"""
import hashlib
import json
import subprocess
from pathlib import Path

NM = "experiments/2026-10-14_q3_new_mains_launch/config"
V4 = ("origin/aswin/q3-overnight-chain-tables",
      "experiments/2026-10-14_q3_overnight_chain/results/hygiene/setup_failures_nm360_v4_2026-10-15.json")
out = Path(__file__).parent / "config" / "tasks_ep2_nm_351.json"
assert not out.exists(), out
show = lambda ref, p: subprocess.run(["git", "show", f"{ref}:{p}"], capture_output=True, check=True, timeout=120).stdout
ref = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
keys, srcs = [], []
for g in ("s1", "s2", "s3", "s4", "s5"):
    raw = show("HEAD", f"{NM}/tasks_nm_{g}_launch.json")
    srcs.append({"path": f"{NM}/tasks_nm_{g}_launch.json", "sha256": hashlib.sha256(raw).hexdigest()})
    keys += [c["key"] for c in json.loads(raw)["cells"] if "real_arms" not in c]
assert len(keys) == 360 == len(set(keys)), len(keys)
v4raw = show(*V4)
v4 = json.loads(v4raw)
excl = sorted(k for ks in v4["cells"].values() for k in ks)
assert v4["v"] == 4 and v4["counts"]["cells"] == len(excl) == 9 and set(excl) <= set(keys), excl
tasks = [k for k in keys if k not in set(excl)]
assert len(tasks) == 351
v4sha = subprocess.run(["git", "rev-parse", "--short=9", V4[0]], capture_output=True, text=True, check=True).stdout.strip()
out.write_text(json.dumps({"origin": "nm360 (S_newmains)", "source": f"the S1-S5 launch files' r1 keys @ {ref} minus setup_failures v4's 9 "
                           f"cells ({V4[0]} @ {v4sha}: {V4[1]}, sha256 {hashlib.sha256(v4raw).hexdigest()})",
                           "inputs": srcs, "excluded": excl, "n": len(tasks), "tasks": tasks}, indent=1) + "\n")
print(out, hashlib.sha256(out.read_bytes()).hexdigest(), len(tasks))
