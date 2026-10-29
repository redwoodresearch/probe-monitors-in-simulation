"""POST HOC, exploratory ($0, numbers only): plan B's S3 (side by side, both orders) on the 50 practice pairs, which
amendment 1 says reports pair accuracy and the first-position "real" rate but the pick stage only counted. From the
stored practice records; S3 can't be picked, so this changes nothing.

Per order: an answer names the transcript (1 or 2) whose outputs the simulator wrote. Accuracy = the share naming the
simulated one; position-balanced accuracy averages the two orders per pair. First-position "real" rate = the share of
answers naming transcript 2 as simulated (so transcript 1 as real), the lean PR #293 reported (64.9%).

  python3 experiments/2026-10-11_claimB_judge_sweep/analyze_POSTHOC_s3_practice.py <practice calls.jsonl>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent


def main(path):
    recs = [json.loads(line) for line in Path(path).read_text().splitlines()]
    s3 = {}
    for redraw in (False, True):
        for r in recs:
            if r["kind"] == "main|S3" and r["status"] == "ok" and ("redraw" in r["replicate"]) == redraw:
                s3.setdefault(r["arm"], r["p"])  # arm: "<cell>|sim-first" or "<cell>|e1-first"
    statuses = {}
    for r in recs:
        if r["kind"] == "main|S3":
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    cells = sorted({a.rsplit("|", 1)[0] for a in s3})
    both = [c for c in cells if f"{c}|sim-first" in s3 and f"{c}|e1-first" in s3]
    right = {c: ((s3[f"{c}|sim-first"] == 1.0) + (s3[f"{c}|e1-first"] == 2.0)) / 2 for c in both}
    answers = [s3[f"{c}|{o}"] for c in both for o in ("sim-first", "e1-first")]
    out = {"label": "POST HOC exploratory: S3 on practice (numbers only)", "statuses": statuses, "pairs_both_orders": len(both),
           "accuracy_sim_first": sum(s3[f"{c}|sim-first"] == 1.0 for c in both) / len(both),
           "accuracy_e1_first": sum(s3[f"{c}|e1-first"] == 2.0 for c in both) / len(both),
           "position_balanced_accuracy": sum(right.values()) / len(both),
           "first_position_real_rate": sum(a == 2.0 for a in answers) / len(answers),
           "inputs": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"s3_practice_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "git", "script_sha256")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
