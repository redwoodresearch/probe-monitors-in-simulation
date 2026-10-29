"""AI-judge planning ($0): the whole job's API price, from `results/judge_tokens_*.json` (tokens read per arm) and the
listed rates. No model is called. Rates (USD per million tokens, input / output):
- GPT-5.6 Sol, direct OpenAI key: 5 / 30 (sce `src/sim/prices.py`, verified there 2026-08-11); Batch API half.
- Claude Fable 5.1, via OpenRouter: 10 / 50; its `:batch` route 5 / 25 (OpenRouter /models/user, read 11:51 PT 10-11
  by check_models.py; not in sim.prices, so the run script registers this row itself).
- GPT-6 Astra, direct OpenAI key: 10 / 50 (sim.prices); Batch API half. Optional third judge only.
Assumptions, each a range: judge-tokenizer tokens per Qwen3-32B token 1.00-1.15; prompt and instructions 1,500 tokens
per call; output (reasoning plus answer) per call 2,000-8,000 for Sol at high effort, 1,000-6,000 for Fable 5.1 and
Astra. The whole job adds: planted copies (60 simulated + 60 real e1 if Aswin adds those), a 100-run test-retest,
redraws for refusals and parse failures (up to 5%), and the dev pilot (20 runs per arm plus 30 planted copies).

  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_judge/analyze_judge_price.py results/judge_tokens_<stamp>.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
RATES = {"sol": (5.0, 30.0), "fable51": (10.0, 50.0), "astra": (10.0, 50.0)}  # standard; batch = half for all three
OUT_TOK = {"sol": (2000, 8000), "fable51": (1000, 6000), "astra": (1000, 6000)}
CONV, PROMPT = (1.00, 1.15), 1500
EXTRA = {"planted": 120, "retest": 100}
REDRAW = 0.05
PILOT_RUNS = 20 * 3 + 30


def cost(judge, calls, mean_read, batch, hi):
    i, o = RATES[judge]
    tin = calls * (mean_read * CONV[hi] + PROMPT)
    tout = calls * OUT_TOK[judge][hi]
    f = 0.5 if batch else 1.0
    return f * (tin * i + tout * o) / 1e6


def main(path):
    src = HERE / path
    tok = json.loads(src.read_text())["rows"]
    test = [tok[f"test|{a}"] for a in ("sim-attack-fixed", "real-attack-e1", "real-attack-e2")]
    dev = [tok[f"dev|{a}"] for a in ("sim-attack-fixed", "real-attack-e1", "real-attack-e2")]
    mean_test = sum(r["read_total"] for r in test) / sum(r["runs"] for r in test)
    mean_dev = sum(r["read_total"] for r in dev) / sum(r["runs"] for r in dev)
    n_full = sum(r["runs"] for r in test)
    out = {"label": "AI-judge planning: whole-job API price (no model called; assumptions in the docstring)", "mean_read_test": mean_test,
           "mean_read_dev": mean_dev, "runs_full": n_full, "rows": {}}
    for scope, n in (("full", n_full), ("reduced_200_per_arm", 600)):
        calls = (n + EXTRA["planted"] + EXTRA["retest"]) * (1 + REDRAW)
        for batch in (False, True):
            row = {}
            for j in ("sol", "fable51", "astra"):
                row[j] = [round(cost(j, calls, mean_test, batch, h), 1) for h in (0, 1)]
            row["sol_plus_fable51"] = [round(row["sol"][h] + row["fable51"][h], 1) for h in (0, 1)]
            out["rows"][f"{scope}|{'batch' if batch else 'standard'}"] = {"calls_per_judge": round(calls), **row}
    pcalls = PILOT_RUNS * (1 + REDRAW)
    out["rows"]["pilot|standard"] = {"calls_per_judge": round(pcalls), **{j: [round(cost(j, pcalls, mean_dev, False, h), 1) for h in (0, 1)] for j in ("sol", "fable51", "astra")}}
    out["rows"]["pilot|standard"]["sol_plus_fable51"] = [round(out["rows"]["pilot|standard"]["sol"][h] + out["rows"]["pilot|standard"]["fable51"][h], 1) for h in (0, 1)]
    pilot = out["rows"]["pilot|standard"]["sol_plus_fable51"]
    for k, v in out["rows"].items():  # the whole job: each scope's main run plus the dev pilot (always at standard rates)
        if not k.startswith("pilot"):
            v["whole_job_sol_plus_fable51_with_pilot"] = [round(v["sol_plus_fable51"][h] + pilot[h], 1) for h in (0, 1)]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "inputs": {src.name: hashlib.sha256(src.read_bytes()).hexdigest()}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"judge_price_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(f"mean tokens read: test {mean_test:.0f}, dev {mean_dev:.0f}; runs {n_full}")
    for k, v in out["rows"].items():
        print(k, v)
    print("->", dst)


if __name__ == "__main__":
    main(sys.argv[1])
