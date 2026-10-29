"""Per T196 task: does its side task have at least one authored attack-strategy file? File COUNTS only; no file is opened.

  uv run --no-project --python 3.12 python experiments/2026-10-14_approach_provenance_09/make_strategy_bits.py \\
      --registry ~/simulated-control-evals-sept/.settings/linuxarena --t196 <pilot_tasks_amendment9_all196.json> --out <json>
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--registry", type=Path, required=True)
p.add_argument("--t196", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
reg_commit = subprocess.run(
    ["git", "-C", str(a.registry), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
cells = json.loads(a.t196.read_text())["cells"]
bits = {}
for c in cells:
    d = a.registry / c["env"] / "side_tasks" / c["side"] / "docs" / "attack_strategies"
    if not (a.registry / c["env"] / "side_tasks" / c["side"]).is_dir():
        raise SystemExit(f"side task folder missing for {c['key']}")
    n = len([f for f in d.glob("*/*.md")]) if d.is_dir() else 0
    bits[c["key"]] = {
        "env": c["env"],
        "side": c["side"],
        "has_strategy": n > 0,
        "n_strategy_files": n,
    }
out = {
    "label": "per T196 task: authored attack-strategy file count of its side task (file counts only)",
    "registry": str(a.registry),
    "registry_commit": reg_commit,
    "t196_sha256": hashlib.sha256(a.t196.read_bytes()).hexdigest(),
    "n_tasks": len(bits),
    "n_has_strategy": sum(b["has_strategy"] for b in bits.values()),
    "tasks": bits,
}
a.out.open("x").write(json.dumps(out, indent=1, sort_keys=True))
print(
    out["n_tasks"],
    "tasks;",
    out["n_has_strategy"],
    "with a strategy; registry",
    reg_commit[:9],
)
