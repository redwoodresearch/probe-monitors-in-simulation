"""How many actions of each simulated run were copied from a real run before the simulator took over (local, free):
each eval's metadata.fork (prefix_actions, after), counted per simulated arm.

  uv run --no-project --python 3.14 python experiments/2026-10-09_q2_onpolicy_discrimination/check_fork_prefix.py <rerun folder>
"""
import collections
import json
import sys
import zipfile
from pathlib import Path

for arm in ("sim_honest@september", "sim_honest@today", "sim_attack@september", "sim_attack@today"):
    c = collections.Counter()
    for f in sorted(Path(sys.argv[1]).glob(f"{arm}/*/*.eval")):
        with zipfile.ZipFile(f) as z:
            ns = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
            fk = json.loads(z.read(ns[0]))["metadata"]["fork"] if ns else {}
        c[f"prefix_actions={fk.get('prefix_actions')}, after={fk.get('after')}" if ns else "no sample"] += 1
    print(arm, dict(c))
