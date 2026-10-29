"""AI-judge pilot: whole-job price as approved before 12:06 PT 10-11 (Aswin via 0d): three judges (GPT-5.6 Sol high,
Claude Fable 5.1, GPT-6 Astra as a labelled extra), simulated and real-e1 plants at three positions, a test-retest subset,
redraws, reasoning tokens. $0: no model is called. Rates, token assumptions and the per-call formula are
analyze_judge_price.py's (imported unchanged); tokens read are the dev-environment mean from results/judge_tokens_*.json.

Specified pilot: 20 runs per arm (fixed sim, real e1, real e2) = 60 runs; plants 10 simulated + 10 real-e1 runs x 3
positions = 60 copies; test-retest 10 runs; 5% redraws. Also: the largest pilot no bigger than specified (even runs per
arm r <= 20, plant runs per arm r / 2, retest r / 2) whose buffered (x1.3) top-end price stays within $125; the cap is
compared on unrounded totals.

  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_judge/analyze_judge_pilot_price.py results/judge_tokens_<stamp>.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_judge_price as P  # noqa: E402  (RATES, OUT_TOK, CONV, PROMPT, REDRAW, cost)

JUDGES, CAP, BUFFER = ("sol", "fable51", "astra"), 125.0, 1.3


def calls(r, p, retest):
    return (3 * r + 2 * 3 * p + retest) * (1 + P.REDRAW)


def price(n_calls, mean_read):
    """(display dict with rounded dollars, unrounded buffered top-end total for the cap)."""
    per = {j: [P.cost(j, n_calls, mean_read, False, h) for h in (0, 1)] for j in JUDGES}
    total = [sum(per[j][h] for j in JUDGES) for h in (0, 1)]
    shown = {j: [round(x, 1) for x in v] for j, v in per.items()}
    return shown | {"total_raw": [round(x, 1) for x in total], "total_buffered": [round(BUFFER * x, 1) for x in total]}, BUFFER * total[1]


def main(path):
    src = HERE / path
    tok = json.loads(src.read_text())["rows"]
    dev = [tok[f"dev|{a}"] for a in ("sim-attack-fixed", "real-attack-e1", "real-attack-e2")]
    mean_dev = sum(r["read_total"] for r in dev) / sum(r["runs"] for r in dev)
    spec = {"runs_per_arm": 20, "plant_runs_per_arm": 10, "retest": 10}
    n = calls(spec["runs_per_arm"], spec["plant_runs_per_arm"], spec["retest"])
    out = {"label": "AI-judge pilot: whole-job price, three judges (no model called)", "mean_read_dev": mean_dev, "cap_buffered": CAP,
           "specified": spec | {"calls_per_judge": round(n, 1)}}
    shown, top = price(n, mean_dev)
    out["specified"] |= shown | {"within_cap": top <= CAP}
    fit = None
    for r in range(20, 1, -2):
        m = calls(r, r // 2, r // 2)
        shown, top = price(m, mean_dev)
        if top <= CAP:
            fit = {"runs_per_arm": r, "plant_runs_per_arm": r // 2, "retest": r // 2, "calls_per_judge": round(m, 1), **shown}
            break
    out["largest_within_cap_at_top_end"] = fit
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "price_script_sha256": hashlib.sha256((HERE / "analyze_judge_price.py").read_bytes()).hexdigest(),
            "inputs": {src.name: hashlib.sha256(src.read_bytes()).hexdigest()}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"judge_pilot_price_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k in ("mean_read_dev", "specified", "largest_within_cap_at_top_end")}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main(sys.argv[1])
