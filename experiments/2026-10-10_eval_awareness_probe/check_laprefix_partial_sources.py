"""Count content test A's prefix sources in the stopped first run's checkpoint ($0; scores and source keys only).

Partial, 94 of 120 pairs, not the pre-registered test: no score is read here. It checks the source assignment only.
For each held-out task covered so far, the prefix source run and its LinuxArena main task (the key's "<task>:<hash>"
part, before the colon), whether any task has two sources, and whether any LinuxArena main task repeats across tasks.

  uv run --no-project python experiments/2026-10-10_eval_awareness_probe/check_laprefix_partial_sources.py --ck .run_cache/eval_awareness_probe/partial/laprefix_ck_shard0.json
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--ck", required=True)
a = ap.parse_args()
raw = (ROOT / a.ck).read_bytes()
ck = json.loads(raw)
src = {}
for r in ck["rows"]:
    if r.get("prefix_source"):
        src.setdefault(r["task"], set()).add(r["prefix_source"])
la_task = {t: [k.split("__")[1].split(":")[0] for k in s] for t, s in src.items()}
la_counts = Counter(x for v in la_task.values() for x in v)
out = {"label": "partial, 94 of 120 pairs, not the pre-registered test; source assignment only, no score read",
       "checkpoint": a.ck, "checkpoint_sha256": hashlib.sha256(raw).hexdigest(), "pairs checkpointed": len(ck["pairs"]),
       "held-out tasks covered": len(src), "tasks with more than one source": sorted(t for t, s in src.items() if len(s) > 1),
       "distinct source runs": len({k for s in src.values() for k in s}),
       "distinct LinuxArena main tasks": len(la_counts), "repeated LinuxArena main tasks": sorted(k for k, n in la_counts.items() if n > 1),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / "partial" / f"laprefix_partial_sources_{stamp}.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
