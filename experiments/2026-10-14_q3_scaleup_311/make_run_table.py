"""The scale-up's run table (RULE.md; Amendments 5 and 7): one row per planned run of every quarter. Real arms run in the
quarter's MAIN launch (both real attacks in the same launch: attack_pair_launch "same"); every simulated attack runs in
the quarter's simulated-only FOLLOW-UP launch, so sim_launch differs from run1_launch on every row (flagged). The 8 cells
the cancelled canary shared are in Q4 with full arms, tagged "same" like every other cell. Keys and counts only.

  python make_run_table.py <out json> <quarter>=<launch tasks json>=<main launch id>=<follow-up launch id>=<host type> ...
"""

import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
assert not out.exists(), f"{out} exists"
rows = []
for spec in sys.argv[2:]:
    q, tasks, main, fu, host = spec.split("=")
    for c in json.loads(Path(tasks).read_text())["cells"]:
        r2 = "real_arms" in c
        base = {"quarter": int(q), "host_instance_type": host, "cell_key": c["key"].removesuffix("_r2"), "cell": c["cell"],
                "env": c["env"], "attack_pair_launch": c["attack_pair_launch"], "empty_approach": c["approach_chars"] == 0,
                "run1_launch": main, "sim_launch": fu, "sim_launch_differs_from_run1": True}
        arms = [("attack", 2, main)] if r2 else [("honest_a", None, main), ("attack", 1, main), ("honest_b", None, main),
                                                 ("sim_attack@september", None, fu)]
        for arm, idx, launch in arms:
            rows.append(base | {"run_key": f"{arm}/{c['key']}", "arm": arm, "attack_index": idx, "launch_id": launch})
out.write_text(json.dumps({"rows": rows}, indent=1))
print(f"{len(rows)} planned runs -> {out} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}; "
      f"attack_pair_launch same {sum(r['attack_pair_launch'] == 'same' for r in rows)}")
