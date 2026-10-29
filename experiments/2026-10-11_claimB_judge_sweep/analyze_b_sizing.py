"""Plan B sizing (PLAN.md amendment 1; $0, laptop, numbers only, no API call): power of the per-judge AUROC rule on the
old judge's 200 attack test cells, and the price of the second judge's parse pilot and of the whole job, per judge.

Power: D = AUROC(judge) - AUROC(probe) on the same 400 runs; P(clear) = P(upper 95% < -0.10) + P(lower 95% > -0.10),
normal approximation, SE = the stand-in's D SE on exactly these cells (Sol S0 vs the probe, from the sizing JSON).
Price: Sol by Batch API from the old run's measured judge tokens per Qwen token and output tokens; Gemini 3.1 Pro
preview at OpenRouter's listed standard rate (judge_models JSON), with a range for its unmeasured tokenizer and output.

  python3 experiments/2026-10-11_claimB_judge_sweep/analyze_b_sizing.py <sweep_sizing.json> <judge_tokens.json> <judge_models.json>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
MARGIN, GAPS = -0.10, [0.0, -0.05, -0.15, -0.20, -0.263, -0.30]
CELLS, N_DEV, N_PLANT, POS, N_RETEST, PQS, SHOTS = 200, 50, 20, 3, 60, 3, 2  # amendment 1's sizes (test plants/retest: the old selection's)
PILOT = {"S0": 4, "S1": 2 * 3, "S2": 2, "S3": 2}  # Gemini parse pilot, practice runs: 4 S0 runs, 2 runs x 3 questions, 2 S2 runs, 1 pair x 2 orders
GEMINI = "google/gemini-3.1-pro-preview"
Phi = NormalDist().cdf


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(sizing, tokens, models):
    sz, tok, mod = (json.loads(Path(x).read_text()) for x in (sizing, tokens, models))
    se = sz["power"]["auroc|by_cells_larger_se"]["se_stand_in"]
    clear = {str(g): Phi((MARGIN - g) / se - 1.96) + Phi((g - MARGIN) / se - 1.96) for g in GAPS}
    passes = {str(g): Phi((MARGIN - g) / se - 1.96) for g in GAPS}
    b = sz["price_sol_batch"]["basis"]
    f, out = b["judge_input_per_qwen_token"], b["mean_output_tokens"]
    t = tok["rows"]
    read = {"practice": (t["dev|sim-attack-fixed"]["read_mean"] + t["dev|real-attack-e1"]["read_mean"]) / 2,
            "test": (t["test|sim-attack-fixed"]["read_mean"] + t["test|real-attack-e1"]["read_mean"]) / 2}
    g = mod["openrouter"][GEMINI]
    judges = {"sol_batch": [(f, out, 2.5, 15.0)],  # $5/$30 per million, Batch API half (the old run's basis)
              "gemini_standard_low": [(f, out, g["prompt_per_M"], g["completion_per_M"])],
              "gemini_standard_high": [(1.15, 8000.0, g["prompt_per_M"], g["completion_per_M"])]}

    def call(j, split, k=1, shots=0):
        fi, o, ri, ro = judges[j][0]
        return (fi * (k * read[split] + shots * read["practice"]) * ri + o * ro) / 1e6

    runs_test = 2 * CELLS + CELLS + 2 * N_PLANT * POS + N_RETEST  # sim + e1, e2 control, planted copies, retest
    price = {}
    for j in judges:
        practice = {"S0": 2 * N_DEV * call(j, "practice"), "S1": 2 * N_DEV * PQS * call(j, "practice"),
                    "S2": 2 * N_DEV * call(j, "practice", shots=SHOTS), "S3_explore": 2 * N_DEV * call(j, "practice", k=2)}
        test = {"S0": runs_test * call(j, "test") * 1.05, "S1": runs_test * PQS * call(j, "test") * 1.05,
                "S2": runs_test * call(j, "test", shots=SHOTS) * 1.05}
        price[j] = {"practice": practice, "practice_total": sum(practice.values()), "test_if_winner": test,
                    "whole_raw": {k: sum(practice.values()) + v for k, v in test.items()}}
    pilot = {j: PILOT["S0"] * call(j, "practice") + PILOT["S1"] * call(j, "practice") + PILOT["S2"] * call(j, "practice", shots=SHOTS)
             + PILOT["S3"] * call(j, "practice", k=2) for j in ("gemini_standard_low", "gemini_standard_high")}
    sol_reuse = price["sol_batch"]["test_if_winner"]["S0"]  # if S0 wins for Sol, the stored 10-11 run is reused
    price["sol_batch"]["whole_raw_net_of_reuse"] = {**price["sol_batch"]["whole_raw"], "S0": price["sol_batch"]["practice_total"]}
    out_ = {"label": "Plan B sizing (plan only; $0; numbers only)", "se_D_auroc_200_cells": se, "p_clear": clear, "p_pass": passes,
            "read_mean_qwen": read, "gemini_rates_per_M": g, "price": price, "gemini_parse_pilot": {"calls": PILOT, "usd": pilot},
            "sol_test_S0_reusable_usd": sol_reuse,
            "inputs": {str(x): sha(x) for x in (sizing, tokens, models)},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout.strip()),
            "script_sha256": sha(__file__)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"b_sizing_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out_) + "\n")
    print(json.dumps(out_, indent=1)[:5000])
    print("->", dst)


if __name__ == "__main__":
    main(*sys.argv[1:4])
