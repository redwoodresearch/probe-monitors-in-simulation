"""Plot the concept probe's gate (far placement) and, if given, stage 5's R per readout. One figure per result.

  uv run --no-project --with matplotlib python experiments/2026-10-11_fake_env_concept_probe/plot_concept.py --gate results/concept_gate_analysis_<stamp>.json [--real results/concept_real_analysis_<stamp>.json]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--gate", required=True)
ap.add_argument("--real", default="")
a = ap.parse_args()
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
g = json.loads((HERE / a.gate).read_text())
fig, ax = plt.subplots(figsize=(7.5, 4.2))
labs = [("far: plant minus neutral", "planted line\nminus neutral line"), ("far: plant minus original", "planted line\nminus nothing"),
        ("far: neutral minus original", "neutral line\nminus nothing")]
w = 0.38
for j, (k, col, name) in enumerate((("stage 4 (content)", "#4C72B0", "content (decides)"), ("stage 4 (turn start)", "#9AA5B1", "turn start"))):
    for i, (key, _) in enumerate(labs):
        v, (lo, hi) = g[k][key]["logits"], g[k][key]["95%"]
        b = ax.bar(i + (j - 0.5) * w, v, w, yerr=[[v - lo], [hi - v]], capsize=4, color=col, label=name if i == 0 else None)
        ax.bar_label(b, labels=[f"{v:+.2f}"], padding=3, fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(3), [x for _, x in labs], fontsize=9)
ax.set_ylabel("run-score change, logits\n(higher = reads more 'simulated')")
ax.set_title(f"The concept probe separates the planted line from a neutral one far from the turn ({g['stage 4']['runs with a matched call']} real runs)", fontsize=10)
ax.legend(fontsize=8)
fig.text(0.5, -0.02, "Line at the end of the system prompt. Matched calls. 95% intervals: 2,000 resamples of runs.", ha="center", fontsize=8)
fig.tight_layout()
out = HERE / "results" / f"concept_gate_figure_{stamp}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"-> {out}")
if a.real:
    r = json.loads((HERE / a.real).read_text())
    K = "R = mean|sim - e1| / mean|e2 - e1|"
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.axhspan(0, 1.2, color="#E5E5E5", label="no more than real vs real (interval wholly at or below 1.2)")
    ax.axhline(1 / 1.2, color="#777777", lw=0.8, ls="--", label="interval wholly below: less than real vs real")
    ax.axhline(1, color="black", lw=0.8)
    for i, (k, name, col) in enumerate((("C", "content (decides)", "#4C72B0"), ("T", "turn start", "#9AA5B1"))):
        v, (lo, hi) = r[k][K], r[k]["R 95%"]
        b = ax.bar(i, v, 0.6, yerr=[[v - lo], [hi - v]], capsize=4, color=col)
        ax.bar_label(b, labels=[f"{v:.2f}"], padding=3, fontsize=9)
    ax.set_xticks(range(2), ["content (decides)", "turn start"], fontsize=9)
    ax.set_ylabel("real-vs-sim gap ÷ real-vs-real gap\n(above the band = sim differs more)")
    ax.set_title(f"{r['cells in the main comparison']} matched cells: {r['C']['verdict (deciding)']}", fontsize=10)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, "Gap = mean |run-score difference| per cell. 95% intervals: 2,000 draws of main tasks.", ha="center", fontsize=8)
    fig.tight_layout()
    out = HERE / "results" / f"concept_real_figure_{stamp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"-> {out}")
