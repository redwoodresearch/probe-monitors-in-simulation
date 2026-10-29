"""Fix (b): write <result>.arms.json beside a fix (b) result file, recording the honest-arm mapping (the analysis scripts
run unchanged, so their outputs label the arms by the relabelled world) and the result file's sha256. Local, $0.

  python stamp_arms.py <result json> [...] [--invalid <result key>=<reason> ...]

--invalid records a block of the result that is not a fix (b) result (e.g. D2's generalization, Addendum 1).
"""

import hashlib
import json
import sys
from pathlib import Path

MAPPING = {"honest_a": "physical arm honest_b (training negatives and the 1% threshold set)",
           "honest_b": "physical arm honest_a (false alarms only; the null gate's negatives)"}

args = sys.argv[1:]
invalid = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--invalid")
files = [a for i, a in enumerate(args) if a != "--invalid" and (i == 0 or args[i - 1] != "--invalid")]
for f in files:
    p = Path(f)
    side = p.with_name(p.name + ".arms.json")
    if side.exists():
        raise SystemExit(f"{side} exists")
    side.write_text(json.dumps({"result": p.name, "result_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                "plan": "experiments/2026-10-11_sim_source_audit/fixes/PLAN_b_honest_swap_POSTHOC.md",
                                "labels_in_result_mean": MAPPING} | ({"invalid_blocks": invalid} if invalid else {}), indent=1))
    print("->", side)
