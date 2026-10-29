"""Plot the distance check: per-token mean paired shift by distance quintile, per placement, as % of the synthetic reference.

  uv run --no-project --with matplotlib python experiments/2026-10-10_eval_awareness_probe/plot_distance.py --result results/distance_<stamp>.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
TITLES = {"far": "Line on the system prompt\n(x: position in the run)", "near": "Line before every calling turn\n(x: tokens from the line)",
          "checker": "Checker comment before every calling turn\n(x: tokens from the comment)"}

ap = argparse.ArgumentParser()
ap.add_argument("--result", required=True)
a = ap.parse_args()
res = json.loads((HERE / a.result).read_text())
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, name in zip(axes, ("far", "near", "checker")):
    qs = res[name]["per_token"]["quintiles"]
    m = [100 * q["mean"] for q in qs]
    err = [[100 * (q["mean"] - q["95%"][0]) for q in qs], [100 * (q["95%"][1] - q["mean"]) for q in qs]]
    bars = ax.bar(range(5), m, yerr=err, capsize=4, color="#4C72B0")
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in m], padding=3, fontsize=9)
    ax.set_xticks(range(5), [f"{q['distance range'][0]:,}–\n{q['distance range'][1]:,}" for q in qs], fontsize=8)
    ax.set_title(f"{TITLES[name]}\n{res[name]['runs']} runs, {res[name]['calls']} calls", fontsize=10)
    ax.set_xlabel("tokens (quintiles of calls)")
axes[0].set_ylabel("shift, % of the synthetic size\n(higher = the probe moves more)")
fig.suptitle("Mean paired shifts are 1–13% of the synthetic size in every distance quintile", fontsize=12)
fig.text(0.5, -0.02, "Bars: mean paired shift per call (inserted minus original; checker: plant minus control), per-token readout. "
         "95% intervals: 2,000 main-task bootstrap draws.",
         ha="center", fontsize=8)
fig.tight_layout()
out = HERE / a.result.replace(".json", ".png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"-> {out}")
