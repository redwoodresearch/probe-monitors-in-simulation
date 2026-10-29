"""Token attribution, counts for RULE.md before any score exists (0d's 13:3x conditions). $0, local CPU, no scores read.

1. The gate's bar: for each of the 30 gutter planted replies, its content tokens n and planted tokens k (gutter-line
   tokens and the final newline, marks.planted_tokens), and the median of the best planted token's rank fraction
   r = (rank - 1) / n under a uniformly random ranking, exactly: P(rank - 1 >= j) = C(n - j, k) / C(n, k). Then the
   gate statistic (np.median of r over the 30 replies) under random ranking, by simulation (100,000 draws, seed below):
   its median and P(statistic <= 0.01).
2. Known-slip marks (marks.known_slips: fixes.py R1 and R2) in the simulated replies: the 91 site replies (the reply
   only) and the 10 whole simulated runs (every output): outputs changed and tokens marked, per mark.
Writes results/marks_counts_<stamp>.json (counts only).

  uv run --no-project --python 3.12 --with transformers==4.56.2 --with jinja2 --with boto3 --with numpy python \\
      experiments/2026-10-13_claimB_token_attribution/count_marks.py
"""

import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "experiments/2026-10-09_fixed_sim")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import marks  # noqa: E402

from probe_shift.models import QWEN3_32B  # noqa: E402

DRAWS, SEED = 100_000, "token-attr-2026-10-13:gate-null"


def best_rank_cdf(n, k):
    """P(rank - 1 <= j), j = 0..n-1, for the best of k planted tokens among n under a random ranking."""
    lc = lambda a, b: math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1) if 0 <= b <= a else -math.inf  # noqa: E731
    return np.array([1 - math.exp(lc(n - j - 1, k) - lc(n, k)) for j in range(n)])


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    staged = HERE / "data/staged.jsonl"
    items = [json.loads(x) for x in staged.read_text().splitlines()]
    reps = []
    for it in (x for x in items if x["kind"] == "plant" and x["slip"] == "gutter"):
        got = marks.planted_tokens(tok, it)
        _, _, spans = marks.layout(tok, it["messages"])
        _, t0, t1 = spans[len(it["messages"]) - 1]
        n, k = t1 - t0, len(got)
        cdf = best_rank_cdf(n, k)
        reps.append({"item": it["item"], "n_content": n, "k_planted": k, "random_median_r": float(np.searchsorted(cdf, 0.5) / n), "cdf": cdf})
    rng = np.random.default_rng(int(hashlib.sha256(SEED.encode()).hexdigest()[:8], 16))
    u = rng.random((DRAWS, len(reps)))
    r = np.stack([np.minimum(np.searchsorted(x["cdf"], u[:, i]), x["n_content"] - 1) / x["n_content"] for i, x in enumerate(reps)], 1)
    stat = np.median(r, axis=1)
    gate = {"replies": len(reps), "n_content": [x["n_content"] for x in reps], "k_planted": [x["k_planted"] for x in reps],
            "per_reply_random_median_r": [x["random_median_r"] for x in reps],
            "random_ranking_gate_statistic_median": float(np.median(stat)), "random_ranking_p_le_0.01": float(np.mean(stat <= 0.01)),
            "draws": DRAWS, "seed": SEED}
    slip = {"site_fake": Counter(), "whole_fake": Counter()}
    for it in items:
        if it["arm"] != "fake":
            continue
        msgs = it["messages"]
        which = {len(msgs) - 1} if it["kind"] == "site" else {i for i, m in enumerate(msgs) if m["role"] == "tool"}
        mk, outs = marks.known_slips(tok, msgs, which)
        c = slip[f"{it['kind']}_fake"]
        c["items"] += 1
        c["items_marked"] += any(mk.values())
        for k_, v in outs.items():
            c[f"outputs_{k_}"] += v
        for k_, v in mk.items():
            c[f"tokens_{k_}"] += len(v)
        c["tokens_marked_any"] += len(set().union(*map(set, mk.values())))
        if it["kind"] == "site":
            _, _, spans = marks.layout(tok, msgs)
            c["reply_content_tokens"] += spans[len(msgs) - 1][2] - spans[len(msgs) - 1][1]
    out = {"label": "token attribution: counts for RULE.md (no scores)", "gate_bar": gate,
           "known_slip_marks": {k: dict(v) for k, v in slip.items()},
           "staged_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(), "marks_sha256": hashlib.sha256((HERE / "marks.py").read_bytes()).hexdigest(),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"marks_counts_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
