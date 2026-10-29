"""POST HOC, exploratory ($0, counts and rates only; 0d's request): 95% intervals clustered by main task for S3 on the
50 practice pairs, beside analyze_POSTHOC_s3_practice.py's point values. From the stored practice records.

Per pair (both orders answered): balanced score = mean of the two orders' correctness (0, 0.5 or 1). Accuracy =
the mean over pairs. First-position "real" rate = the share of all answers naming transcript 2 as simulated. Intervals:
percentile 95% of 2,000 draws resampling main tasks with replacement, default_rng(20261012).

  python3 experiments/2026-10-11_claimB_judge_sweep/analyze_POSTHOC_s3_ci.py <practice calls.jsonl>
"""

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent


def main(path):
    recs = [json.loads(line) for line in Path(path).read_text().splitlines()]
    s3 = {}
    for redraw in (False, True):
        for r in recs:
            if r["kind"] == "main|S3" and r["status"] == "ok" and ("redraw" in r["replicate"]) == redraw:
                s3.setdefault(r["arm"], r["p"])
    pairs = sorted(c for c in {a.rsplit("|", 1)[0] for a in s3} if f"{c}|sim-first" in s3 and f"{c}|e1-first" in s3)
    ok_sf = {c: s3[f"{c}|sim-first"] == 1.0 for c in pairs}
    ok_ef = {c: s3[f"{c}|e1-first"] == 2.0 for c in pairs}
    first_real = {c: (s3[f"{c}|sim-first"] == 2.0) + (s3[f"{c}|e1-first"] == 2.0) for c in pairs}  # answers naming 2 as simulated
    by = defaultdict(list)
    for c in pairs:
        by[c.split(":")[0]].append(c)
    tasks = sorted(by)
    rng, acc, lean = np.random.default_rng(20261012), [], []
    for _ in range(2000):
        pick = [c for t in rng.choice(len(tasks), len(tasks)) for c in by[tasks[t]]]
        acc.append(np.mean([(ok_sf[c] + ok_ef[c]) / 2 for c in pick]))
        lean.append(sum(first_real[c] for c in pick) / (2 * len(pick)))
    out = {"label": "POST HOC exploratory: S3 on practice, task-clustered 95% intervals (counts and rates only)",
           "pairs_both_orders": len(pairs), "main_tasks": len(tasks),
           "correct_sim_first": int(sum(ok_sf.values())), "correct_e1_first": int(sum(ok_ef.values())),
           "position_balanced_accuracy": float(np.mean([(ok_sf[c] + ok_ef[c]) / 2 for c in pairs])),
           "accuracy_ci95": [float(x) for x in np.percentile(acc, [2.5, 97.5])],
           "answers": 2 * len(pairs), "answers_naming_transcript_2": int(sum(first_real.values())),
           "first_position_real_rate": float(sum(first_real.values()) / (2 * len(pairs))),
           "first_position_real_ci95": [float(x) for x in np.percentile(lean, [2.5, 97.5])],
           "inputs": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"s3_ci_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "git", "script_sha256")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
