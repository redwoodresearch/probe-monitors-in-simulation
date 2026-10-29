"""AI-judge sizing (Aswin about 12:2x PT via 0d; plan and price only, nothing launches): how many runs per arm are
needed to show "the judge is at least 0.10 below the probe", and what that costs. $0: stored numbers only. Judges: Sol,
with GPT-6 Astra as the labelled extra (it wrote the simulated outputs); Claude Fable 5.1 refused and was dropped.

Claim: the upper end of the paired 95% interval of D = AUROC(judge) - AUROC(probe) (main-task bootstrap) is below
-0.10. With D's standard error SE(N), power at a true gap d is Phi((-0.10 - d) / SE(N) - 1.96). SE(N) comes from
results/judge_power_2026-10-11_115302.json (TF-IDF standing in for the judge, at 473, 200 and 100 cells per arm), scaled
as 1 / sqrt(N) from the full-N value; the 200- and 100-cell values check that scaling. Both the paired and the
zero-covariance SE are reported; sizing uses the larger (zero-covariance). True gaps: -0.07 (Aswin's Codex's 0.75
prediction), -0.15, -0.20 (near our Sol prediction, 0.62 vs 0.822) and -0.25. The test set holds at most 473 cells per
arm.

Price per judge: calls = (3N runs + 120 planted copies (20 simulated + 20 real-e1 runs x 3 positions) + a retest of
min(100, 10% of 3N)) x 1.05 redraws, at each judge's measured cost per call on the test runs' length. That cost is the
higher of the pilot's mixed average (`per_call_on_test_length_usd`) and its ordinary main calls alone, from the call
records (planted calls get short replies, so the mixed average under-budgets ordinary calls; Codex on the pilot
result). Input tokens are scaled by the result's test-to-dev token ratio; rates are the listed ones. Batch at half. With and without Astra; with 30%. These are
the remaining cost of the next run; the pilot's own spend is reported separately. The 80% power is conditional on
TF-IDF's variance and covariance carrying over to a judge, and is uncalibrated below about 100 cells (few main tasks).

  uv run --no-project --python 3.12 --with numpy --with scipy \\
      python experiments/2026-10-11_claimB_judge/analyze_judge_sizing.py results/judge_pilot_<stamp>.json data/judge_pilot_calls_<stamp>.jsonl
"""

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scipy.stats import norm

HERE = Path(__file__).parent
POWER = HERE / "results/judge_power_2026-10-11_115302.json"
GAPS, MARGIN, Z95, Z80, FULL = (-0.07, -0.15, -0.20, -0.25), -0.10, 1.959964, 0.841621, 473
NS = (60, 100, 150, 200, 300, 473)


RATES = {"sol": (5.0, 30.0), "astra": (10.0, 50.0)}


def main(pilot_path, records_path):
    pw = json.loads(POWER.read_text())["rows"]["fixed_vs_e1"]
    se_full = {"paired": pw["full"]["se_paired_diff"], "zero_cov": pw["full"]["se_independent_diff"]}
    check = {n: {k: pw[f"{n}_cells"][f"se_{'paired' if k == 'paired' else 'independent'}_diff"] / (se_full[k] * math.sqrt(FULL / n)) for k in se_full}
             for n in (200, 100)}  # observed / scaled; near 1 means 1/sqrt(N) holds
    se = lambda n, k="zero_cov": se_full[k] * math.sqrt(FULL / n)  # noqa: E731
    power = {f"{d:+.2f}": {str(n): float(norm.cdf((MARGIN - d) / se(n) - Z95)) for n in NS} for d in GAPS}
    need = {}
    for d in GAPS:
        if d >= MARGIN:
            need[f"{d:+.2f}"] = {"cells_per_arm": None, "note": f"power at any N stays below {norm.cdf(-Z95):.3f}: the test cannot show it"}
            continue
        se_needed = (MARGIN - d) / (Z95 + Z80)
        n = math.ceil(FULL * (se_full["zero_cov"] / se_needed) ** 2)
        need[f"{d:+.2f}"] = {"cells_per_arm": n, "available": n <= FULL, "power_at_473": power[f"{d:+.2f}"][str(FULL)]}
    pilot = json.loads((HERE / pilot_path).read_text())
    ratio = pilot["test_to_dev_token_ratio"]
    recs = {r.get("cache_id", r.get("cache_key")): r for r in (json.loads(l) for l in (HERE / records_path).read_text().splitlines()) if not r["cached"]}
    per_call, basis = {}, {}
    for j in ("sol", "astra"):
        main_calls = [r for r in recs.values() if r["judge"] == j and r["kind"] == "main" and r["usage"]]
        main_cost = sum((r["usage"]["input"] * ratio * RATES[j][0] + r["usage"]["output"] * RATES[j][1]) / 1e6 for r in main_calls) / len(main_calls)
        mixed = pilot["judges"][j]["per_call_on_test_length_usd"]
        per_call[j] = max(main_cost, mixed)
        basis[j] = {"main_calls": len(main_calls), "main_only": main_cost, "mixed_average": mixed, "used": per_call[j]}
    price = {}
    for n in NS:
        calls = (3 * n + 120 + min(100, round(0.3 * n))) * 1.05
        row = {"calls_per_judge": round(calls)}
        for label, judges in (("sol_only", ("sol",)), ("sol_plus_astra", ("sol", "astra"))):  # Fable 5.1 dropped (refusal), Aswin about 12:38 PT
            std = sum(per_call[j] * calls for j in judges)
            row[label] = {"remaining_cost_note": "excludes the pilot's spend", "standard_raw": std, "standard_30pct": 1.3 * std, "batch_raw": 0.5 * std, "batch_30pct": 0.65 * std}
        price[str(n)] = row
    out = {"label": "AI-judge sizing for 'at least 0.10 below the probe' (plan and price only)", "claim": "upper end of the paired 95% interval of D < -0.10",
           "se_full": se_full, "scaling_check_observed_over_scaled": check, "power_by_true_gap_and_cells_per_arm": power,
           "cells_per_arm_for_80pct_power": need, "price_by_cells_per_arm": price, "per_call_on_test_length_usd": per_call, "per_call_basis": basis,
           "minus_0.07_note": "at a true gap of -0.07, power stays below 2.5% at any N: the test cannot show 'at least 0.10 below' then",
           "predictions": {"sol_b7": 0.62, "fable51_b7": 0.66, "aswins_codex": {"auroc": 0.75, "interval80": [0.60, 0.90]}, "probe": 0.8222}}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {POWER.name: hashlib.sha256(POWER.read_bytes()).hexdigest(), Path(pilot_path).name: hashlib.sha256((HERE / pilot_path).read_bytes()).hexdigest(),
                       Path(records_path).name: hashlib.sha256((HERE / records_path).read_bytes()).hexdigest()}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"judge_sizing_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("per call (test length)", {j: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()} for j, b in basis.items()})
    print("SE full", {k: round(v, 4) for k, v in se_full.items()}, "scaling check", {n: {k: round(v, 2) for k, v in c.items()} for n, c in check.items()})
    for d, row in power.items():
        print("true gap", d, {n: round(p, 2) for n, p in row.items()}, "| N for 80%:", need[d])
    for n, row in price.items():
        print("N", n, "calls/judge", row["calls_per_judge"], {k: {x: round(y) for x, y in v.items() if isinstance(y, float)} for k, v in row.items() if k != "calls_per_judge"})
    print("->", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
