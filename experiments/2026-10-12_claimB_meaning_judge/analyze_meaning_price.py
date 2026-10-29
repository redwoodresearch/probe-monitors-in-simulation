"""Meaning judge: price and power before anything runs (PLAN.md; $0, numbers only, no API call).

Calls (PLAN.md): at the strict same-state sites, (e1, e2) and (e1, sim) in both orders; at the Oct-6 honest sites,
(e1, sim) in both orders; 40 planted and 20 identical-control pairs in both orders; 30 retests; a 10-call parse pilot;
+5% redraws. Input per call = (2 replies + 2,000 characters of prompt and command) / characters per token (3.0-4.5,
assumed); output = the measured mean of the closest stored set-up (Sol: plan B's side-by-side S3; Gemini: its parse
pilot; Fable: Sol's, unmeasured). Simulator (unrepaired; v2: only sites without the n437 run's own matched reply): the
Oct-6 run's $90.48 for 200 calls, max $1.43 a call.
Power: per-site agreement difference, D = agree(sim, e1) - agree(e2, e1); SE from the binomial variances of both
arms, n = the strict sites; margin 0.10; P(pass) = P(90% lower > -0.10), P(fail) = P(95% upper < -0.10).

  python3 experiments/2026-10-12_claimB_meaning_judge/analyze_meaning_price.py <prefix_matched.json> <sol_b_calls.jsonl> <gemini_pilot_calls.jsonl>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist, mean
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
Phi, X = NormalDist().cdf, 0.10
SIM_MEAN, SIM_MAX = 90.48 / 200, 1.43  # experiments/2026-10-11_claimB_k_replies/PLAN.md:19 @ ef406960 (the Oct-6 run)
JUDGES = {"gpt-5.6-sol (OpenAI Batch API)": (2.5, 15.0, "S3"), "google/gemini-3.1-pro-preview (OpenRouter batch)": (1.0, 6.0, "gemini"),
          "anthropic/claude-fable-5.1 (OpenRouter batch)": (5.0, 25.0, "S3x2")}  # $ per M in, out; output basis


def main(prefix_path, sol_path, gem_path):
    pm = json.loads(Path(prefix_path).read_text())
    strict = pm["strict_sites"]
    oct6 = [p for p in json.loads(subprocess.run(["git", "show", "3820d9e5:experiments/2026-10-06_matched_step_pairs/results/pairs_2026-10-06_175844.json"],
                                                 capture_output=True, text=True, check=True, cwd=HERE).stdout)["pairs"] if p["arm"] == "honest"]
    s3 = [json.loads(line)["usage"]["output"] for line in Path(sol_path).read_text().splitlines() if '"main|S3"' in line]
    gem = [json.loads(line)["usage"]["completion"] for line in Path(gem_path).read_text().splitlines()]
    out_tok = {"S3": mean(s3), "gemini": mean(gem), "S3x2": 2 * mean(s3)}
    chars = [2 * max(s["reply_chars"]) + 2000 for s in strict] * 4 + [2 * p["real_output_chars"] + 2000 for p in oct6] * 2
    n_calls = {"strict sites, (e1,e2) and (e1,sim), both orders": 4 * len(strict), "Oct-6 sites, (e1,sim), both orders": 2 * len(oct6),
               "planted 40 + identical control 20, both orders": 120, "retest": 30, "parse pilot": 10}
    total_calls = sum(n_calls.values()) * 1.05
    price = {}
    for name, (rin, rout, basis) in JUDGES.items():
        lo_hi = []
        for cpt in (4.5, 3.0):
            per_in = mean(chars) / cpt
            lo_hi.append(total_calls * (per_in * rin + out_tok[basis] * rout) / 1e6)
        price[name] = {"usd_raw": lo_hi, "usd_with_30pct": [1.3 * v for v in lo_hi]}
    new = sum(not s.get("sim_matched") for s in strict)  # v2: sites without the n437 run's own matched reply (unrepaired simulator)
    sims = {"natural_matched": len(strict) - new, "calls": new, "usd_raw_mean": new * SIM_MEAN, "usd_raw_worst": new * SIM_MAX}
    power = {}
    for n, pr, d in ((n, pr, d) for n in (len(strict), sims["natural_matched"]) for pr in (0.02, 0.10) for d in (0.0, 0.05, 0.10, 0.20, 0.30)):
        se = ((pr * (1 - pr) + (pr + d) * (1 - pr - d)) / n) ** 0.5
        power[f"{n} sites, real disagree {pr}, sim worse by {d}"] = {"se": se, "p_pass": Phi((X - d) / se - 1.645), "p_fail": Phi((d - X) / se - 1.96)}
    out = {"label": "Meaning judge: price and power (plan only; $0)", "strict_sites": len(strict), "oct6_honest_sites": len(oct6),
           "calls": n_calls, "calls_with_redraws": total_calls, "output_tokens_basis": out_tok, "mean_input_chars": mean(chars),
           "judge_price": price, "simulator": sims, "power": power,
           "inputs": {Path(x).name: hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in (prefix_path, sol_path, gem_path)},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"meaning_price_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs",)}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
