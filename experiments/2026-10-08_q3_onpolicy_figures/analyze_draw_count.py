"""How many refit draws the 196-task scoring needs: the Monte Carlo error of the 95% interval's endpoints against the
number of draws B. Local, free; reads a stored refit-draw file (Figure 3's 2,000 draws at 20 tasks).

For B in BS, REPS sets of B draws are taken with replacement from the stored draws (seed 20261010), standing in for B
fresh draws; each set's 2.5 and 97.5 percentiles are its interval. Reported per measure (real minus simulated, the
real-trained probe): the endpoints' standard deviation across sets, as a share of the stored interval's half-width, so
it carries over to another task count if the draws' distribution keeps its shape; beside it the normal-theory value
√(p(1 − p)/B) / φ(z_p) / z_p. The planted checks (a), (a′), (b) and the D plants use fixed-score resamples, not refit
draws, so B does not touch them.

  python3 experiments/2026-10-08_q3_onpolicy_figures/analyze_draw_count.py <draws file>
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
BS, REPS, SEED = (100, 200, 500, 1000, 2000), 2000, 20261010
MEASURES = {"TPR at 1% (frozen threshold)": "catch_frozen", "FPR at the frozen threshold": "false_alarm_frozen", "AUROC": "auroc"}


def main(path: str):
    r = json.loads(Path(path).read_text())
    boot = [d for d in r["draws"] if d["draw"] > 0]
    rng = np.random.default_rng(SEED)
    theory = lambda B: float(np.sqrt(0.025 * 0.975 / B) / (np.exp(-1.96**2 / 2) / np.sqrt(2 * np.pi)) / 1.96)
    out = {"input": Path(path).name, "stored_draws": len(boot), "reps": REPS, "seed": SEED, "measures": {}}
    for name, k in MEASURES.items():
        g = np.array([d["cells"]["real->real"]["values"][k] - d["cells"]["real->sim"]["values"][k] for d in boot
                      if not d["cells"]["real->real"].get("failed") and not d["cells"]["real->sim"].get("failed")])
        lo, hi = np.percentile(g, [2.5, 97.5])
        half = (hi - lo) / 2
        m = {"usable_draws": len(g), "interval_all_draws": [round(float(lo), 4), round(float(hi), 4)], "half_width": round(float(half), 4),
             "endpoint_sd_share_of_half_width": {}}
        for B in BS:
            e = np.percentile(g[rng.integers(0, len(g), (REPS, B))], [2.5, 97.5], axis=1)
            m["endpoint_sd_share_of_half_width"][B] = {"lower": round(float(e[0].std() / half), 3), "upper": round(float(e[1].std() / half), 3),
                                                       "normal_theory": round(theory(B), 3)}
        out["measures"][name] = m
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"draw_count_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main(sys.argv[1])
