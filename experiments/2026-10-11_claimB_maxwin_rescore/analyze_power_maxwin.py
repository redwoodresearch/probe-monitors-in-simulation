"""Power check for the max-of-rolling-means re-score (PLAN.md), $0 and local. Inputs are committed or S3 results only,
read by key. No new scores are computed.

1. Catch at 1% and AUROC, paired differences between two summaries on the same 473 + 473 test runs. The SE is taken
   from the nearest existing paired result: H2's best-found minus step A (2,000 main-task draws) at 267e438a. The
   minimum detectable gain at 80% power (one-sided 2.5%) is 2.80 x SE. For a +5-point catch gain, the factor by which
   the run count would have to grow is (2.80 x SE / 5)^2.
2. Planted check 1 (150 pairs, bar 0.95). Hanley-McNeil SE at the true AUROC, times the design effect implied by step
   A's observed interval (0.853 [0.8195, 0.8863], tpr_at_fpr). Probability that the point estimate is at least the
   bar and the 95% lower end is above 0.5: one tail above max(bar, 0.5 + 1.96 SE), normal approximation. The rule's
   third condition (above the selection null's 95th percentile, 0.53-0.55 in step A) is not modelled.
All of these are conditional planning estimates (sensitivity calculations): H2's paired difference is between two
different probes, whose covariance can differ from two summaries of one probe, and more runs means more independent
main tasks of the same mix.

  uv run --no-project --python 3.12 python experiments/2026-10-11_claimB_maxwin_rescore/analyze_power_maxwin.py \\
      experiments/2026-10-09_fixed_sim/results/tpr_at_fpr_2026-10-10_235259.json
"""

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
H2_COMMIT, H2_PATH = "267e438a", "experiments/2026-10-11_claimB_h2_probe_best_found/results/h2_summary_2026-10-11_132926.json"
Z = NormalDist()


def hm_se(a: float, n1: int, n2: int) -> float:
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2))


def main():
    raw = subprocess.run(["git", "show", f"{H2_COMMIT}:{H2_PATH}"], capture_output=True, check=True, cwd=HERE).stdout
    h2 = json.loads(raw)["rows"]["fixed_vs_e1"]
    tpr_path = Path(sys.argv[1])
    tpr_raw = tpr_path.read_bytes()
    rows = {r["id"]: r for r in json.loads(tpr_raw)["rows"]}
    k = 1.959964 + 0.841621  # one-sided 2.5% test, 80% power
    out = {"label": "power check for the max-of-rolling-means plan (numbers only)", "inputs": {
        f"{H2_COMMIT}:{H2_PATH}": hashlib.sha256(raw).hexdigest(), str(tpr_path): hashlib.sha256(tpr_raw).hexdigest()}}
    for name, lo, hi, unit in (("catch_1pct", *h2["catch_1pct"]["best_minus_A0_ci95"], 100),
                               ("catch_5pct", *h2["catch_5pct"]["best_minus_A0_ci95"], 100),
                               ("auroc", *h2["D"]["ci95"], 1)):
        se = (hi - lo) / (2 * 1.959964) * unit
        m = 5 if unit == 100 else 0.02  # the plan's "not higher by" margin
        out[name] = {"paired_ci95_source": [lo, hi], "se": se, "mde80": k * se, "margin": m, "n_factor_for_gain_at_margin": (k * se / m) ** 2,
                     "p_higher_by_true_gain": {str(g): 1 - Z.cdf((1.959964 * se - g) / se) for g in (0, m, 2 * m, k * se)},
                     "p_not_higher_by_margin_if_true_gain_0": Z.cdf((m - 1.959964 * se) / se)}
    a = rows["step A|level1 · run mean|plant1|test 5"]
    n1, n2 = a["positive_runs"], a["negative_runs"]
    deff = ((a["auroc_ci95"][1] - a["auroc_ci95"][0]) / (2 * 1.959964)) / hm_se(a["auroc"], n1, n2)
    out["plant1"] = {"observed": a["auroc"], "ci95": a["auroc_ci95"], "pairs": n1, "design_effect_se": deff, "p_detect_point_ge_bar": {}}
    for true in (0.90, 0.93, 0.95, 0.97, 0.99):
        se = hm_se(true, n1, n2) * deff
        p = 1 - Z.cdf((max(0.95, 0.5 + 1.959964 * se) - true) / se)
        out["plant1"]["p_detect_point_ge_bar"][str(true)] = {"se": se, "p": p}
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"power_maxwin_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "git", "script_sha256")}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
