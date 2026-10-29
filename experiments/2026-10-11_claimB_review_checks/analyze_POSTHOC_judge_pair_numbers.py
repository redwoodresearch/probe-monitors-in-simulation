"""POST HOC count for review thread #516 PRRT_kwDOU9RNXc6rN8m0 (4f, 2026-10-11): in every saved describe-judge answer
(experiments/2026-10-09_fixed_sim/data/describe_judge_*.json, structured and prose-parsed), how many pair numbers are
repeated or outside 1-20, in the per-pair picks and in each listed difference's pairs. A repeat would overwrite a pick
(run_describe_judge.py builds {pair: pick}) and an out-of-range number would count as answered. Counts only; prints
no model text. Local and free.

  python3 experiments/2026-10-11_claimB_review_checks/analyze_POSTHOC_judge_pair_numbers.py
"""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
DATA = HERE.parent / "2026-10-09_fixed_sim" / "data"


def main():
    out, total = {"label": "POST HOC (#516 review): repeated or out-of-range pair numbers in saved judge answers; counts only", "files": {}}, Counter()
    for f in sorted(DATA.glob("describe_judge_*.json")):
        d = json.loads(f.read_text())
        rows = []
        for a in d["answers"]:
            ans = a["answer"]
            picks = [p["pair"] for p in ans["pairs"]]
            c = Counter(picks)
            diff_pairs = [q for x in ans.get("differences", []) for q in x.get("pairs", [])]
            dup_within_diff = sum(len(x.get("pairs", [])) - len(set(x.get("pairs", []))) for x in ans.get("differences", []))
            row = {"seed": a["seed"], "format": a.get("format", "json"), "picks": len(picks),
                   "repeated_pick_numbers": sum(v - 1 for v in c.values() if v > 1),
                   "out_of_range_picks": sum(1 for q in picks if not 1 <= q <= 20),
                   "difference_pair_refs": len(diff_pairs), "out_of_range_difference_refs": sum(1 for q in diff_pairs if not 1 <= q <= 20),
                   "repeated_within_a_difference": dup_within_diff}
            rows.append(row)
            total.update({k: v for k, v in row.items() if k not in ("seed", "format")})
        out["files"][f.name] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "answers": rows}
    out["totals"] = dict(total)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"judge_pair_numbers_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(f"{len(out['files'])} files, {sum(len(v['answers']) for v in out['files'].values())} answers; totals: {out['totals']}")
    print("->", dst)


if __name__ == "__main__":
    main()
