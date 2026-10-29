"""Plot retrain A and B's result: each readout's paired shift as a fraction of its own synthetic size, per placement.

  uv run --no-project --with matplotlib python experiments/2026-10-10_eval_awareness_probe/plot_retrain.py --result results/retrain_analysis_<stamp>.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--result", required=True)
a = ap.parse_args()
res = json.loads((HERE / a.result).read_text())
names = {"old_per_token": "Content,\noriginal probe", "A_per_token": "Content,\nretrained (A)", "B": "Turn start\n(B)"}
titles = {"far": "Eval line at the end of the system prompt", "near": "Eval line before every calling turn",
          "checker": "Checker comment vs control comment"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, place in zip(axes, titles):
    m = [100 * res[place][r]["fraction of reference"] for r in names]
    lo = [100 * res[place][r]["fraction 95%"][0] for r in names]
    hi = [100 * res[place][r]["fraction 95%"][1] for r in names]
    bars = ax.bar(range(3), m, yerr=[[v - l for v, l in zip(m, lo)], [h - v for v, h in zip(m, hi)]], capsize=4,
                  color=["#9AA5B1", "#9AA5B1", "#4C72B0"])
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in m], padding=3, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(range(3), list(names.values()), fontsize=8)
    ax.set_title(titles[place], fontsize=10)
axes[0].set_ylabel("paired shift, % of the probe's synthetic size\n(log scale; higher = responds more)")
fig.suptitle("Retraining with real commands (A) changes little; the turn-start readout (B) responds far more", fontsize=11)
fig.text(0.5, -0.02, "99 real honest LinuxArena runs (checker: 20). 95% intervals: 2,000 draws of LinuxArena and synthetic tasks.",
         ha="center", fontsize=8)
fig.tight_layout()
out = HERE / a.result.replace(".json", ".png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"-> {out}")
