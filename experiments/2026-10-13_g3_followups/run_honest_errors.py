"""G3 follow-ups (PLAN.md (1d); POST HOC): per-output Inspect error flags of the honest cohorts, metadata only, as
../2026-10-06_q1_segment_end/run_tool_errors.py made them for the attack cohorts (results/tool_errors_2026-10-06_134908.jsonl
on main): one row per tool output, keyed as the extractions key runs, indexed as reader_messages indexes outputs, with the
Inspect error type or null. No text is written. Streams the two honest cohorts' eval logs from S3 on the laptop with the
user's own AWS session (real honest n1000, 113 MB; Astra simulated honest n437, 120 MB); nothing goes to a container.

  AWS_PROFILE=redwood-sandbox PYTHONPATH=src uv run --no-project --python 3.12 --with boto3 --with pyyaml \\
      python experiments/2026-10-13_g3_followups/run_honest_errors.py
"""
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.transcripts import eval_files, file_samples, tool_errors

HERE = Path(__file__).resolve().parent
COHORTS = {"real-honest": "s3://rr-sce-sim-transcripts/transcripts/2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000/",
           "sim-honest": "la_public10_sim_astra_honest_n437"}

for f in (Path(__file__), HERE / "PLAN.md"):
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
            or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
        raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
rows, summary = [], {}
for arm, alias in COHORTS.items():
    files = eval_files(alias)
    with ThreadPoolExecutor(32) as pool:
        for f, samples in zip(files, pool.map(lambda f: file_samples(alias, f), files)):
            for s in samples:
                key = f"{f.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}"
                rows += [{"arm": arm, "key": key} | r for r in tool_errors(s)]
    mine = [r for r in rows if r["arm"] == arm]
    summary[arm] = {"files": len(files), "runs": len({r["key"] for r in mine}), "tool outputs": len(mine),
                    "errored outputs by type": dict(Counter(r["error"] for r in mine if r["error"]))}
    print(arm, summary[arm], flush=True)
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
out = HERE / "results" / f"honest_errors_{stamp}.jsonl"
out.parent.mkdir(exist_ok=True)
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
(HERE / "results" / f"honest_errors_{stamp}_record.json").write_text(json.dumps({"git": head, "cohorts": COHORTS, "summary": summary, "rows file": out.name}, indent=1) + "\n")
print(f"{len(rows)} tool outputs -> {out}")
