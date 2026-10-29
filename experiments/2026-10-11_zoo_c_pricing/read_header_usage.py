"""Sol's usage per real honest run, from the logs' headers only (no messages): a seeded sample of the public-10 real
honest cohort's overlap800 logs. Writes numbers and file names only.

  aws s3 ls s3://rr-sce-sim-transcripts/transcripts/2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000/logs/ | awk '{print $4}' > logs_list.txt
  AWS_PROFILE=redwood-sandbox uv run --no-project --python 3.12 --with inspect_ai --with s3fs python experiments/2026-10-11_zoo_c_pricing/read_header_usage.py logs_list.txt 60
"""
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from inspect_ai.log import read_eval_log

P = "s3://rr-sce-sim-transcripts/transcripts/2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000/logs/"
SEED = 20261011
with open(sys.argv[1]) as f:
    names = sorted(l.strip() for l in f if l.startswith("overlap800"))
pick = random.Random(SEED).sample(names, int(sys.argv[2]))
out = []
for n in pick:
    h = read_eval_log(P + n, header_only=True)
    mu = (h.stats.model_usage or {}).get("openai/gpt-5.6-sol")
    t = (datetime.fromisoformat(h.stats.completed_at) - datetime.fromisoformat(h.stats.started_at)).total_seconds()
    out.append({"file": n, "samples": h.results.total_samples, "input": mu.input_tokens, "cache_read": mu.input_tokens_cache_read or 0,
                "output": mu.output_tokens, "reasoning": mu.reasoning_tokens or 0, "wall_s": t, "models": sorted(h.stats.model_usage)})
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = Path(__file__).parent / "results" / f"header_usage_{stamp}.json"
path.write_text(json.dumps({"prefix": P, "seed": SEED, "of": len(names), "runs": out}) + "\n")
print(len(out), "->", path)
