"""Plot real against fixed-sim: the directional concordance difference dc per readout, beside the planted far eval line's.

  uv run --no-project --with matplotlib python experiments/2026-10-11_eval_awareness_real_vs_sim/plot_real_vs_sim.py --result results/real_vs_sim_analysis_<stamp>.json --planted results/planted_power_<stamp>.json
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
ap.add_argument("--result", required=True)
ap.add_argument("--planted", required=True)
a = ap.parse_args()
res, pl = (json.loads((HERE / f).read_text()) for f in (a.result, a.planted))
names = {"B": "Turn start (B)", "A": "Content,\nretrained (A)", "old": "Content,\noriginal probe"}
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.axhspan(-0.05, 0.05, color="#E5E5E5", label="within ±0.05 (the margin)")
ax.axhline(0, color="black", lw=0.8)
w = 0.27
for i, r in enumerate(names):
    for j, (v, lo_hi, col, lab) in enumerate((
            (res[r]["dc = P(sim > e1) - P(e2 > e1)"], res[r]["dc 95%"], "#4C72B0", "simulated vs real, against real run e1 (pre-registered)"),
            (pl[r]["observed dc against e2 = P(sim > e2) - P(e1 > e2)"], pl[r]["observed dc against e2 95%"], "#8DA0CB", "simulated vs real, against real run e2"),
            (pl[r]["dc_planted = P(e1 far > e2) - P(e1 > e2)"], pl[r]["dc_planted 95%"], "#DD8452", "real with an explicit eval line far away, against e2"))):
        x = i + (j - 1) * w
        bar = ax.bar(x, v, w, yerr=[[v - lo_hi[0]], [lo_hi[1] - v]], capsize=4, color=col, label=lab if i == 0 else None)
        ax.bar_label(bar, labels=[f"{v:+.3f}"], padding=3, fontsize=8)
ax.set_xticks(range(3), list(names.values()), fontsize=9)
ax.set_ylabel("share of cells where the reading is higher,\nminus the real-vs-real share (0 = no difference)")
ax.set_title(f"By the pre-registered rule, no readout shifts beyond real vs real ({res['cells in the main comparison']} cells);\n"
             "against the other real run, the turn start reads sim slightly higher", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
fig.text(0.5, -0.02, "Claim-B attack-mode runs, Qwen3.5-27B reader. 95% intervals: 2,000 draws of main tasks.", ha="center", fontsize=8)
fig.tight_layout()
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
out = HERE / "results" / f"real_vs_sim_figure_{stamp}.png"  # results/ is append-only: a new name per render
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"-> {out}")
