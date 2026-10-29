"""Per-output tool-error table for the attack cohorts, keyed as the segment-end extraction keys outputs.

One row per tool output: arm, key, step, action, tool, error (the Inspect error type, or null). Used for the
"no tool-error outputs" slice (real runs show argument-validation, timeout and other errors the simulator never
produces). Streams from S3 (network-bound), writes results/tool_errors_<stamp>.jsonl.

  AWS_PROFILE=redwood-sandbox uv run python experiments/2026-10-06_q1_segment_end/run_tool_errors.py
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.transcripts import eval_files, file_samples, tool_errors

HERE = Path(__file__).parent
COHORTS = {"real-attack-e1": "la_public10_real_attack_n1000", "real-attack-e2": "la_public10_real_attack_reroll_n1000",
           "sim-attack": "la_public10_sim_astra_attack_n1000"}

rows = []
for arm, alias in COHORTS.items():
    with ThreadPoolExecutor(32) as pool:
        for f, samples in zip(eval_files(alias), pool.map(lambda f: file_samples(alias, f), eval_files(alias))):
            for s in samples:
                key = f"{f.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}"
                rows += [{"arm": arm, "key": key} | r for r in tool_errors(s)]
    print(arm, dict(Counter(r["error"] for r in rows if r["arm"] == arm and r["error"])), flush=True)
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
out = HERE / "results" / f"tool_errors_{stamp}.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"{len(rows)} tool outputs -> {out}")
