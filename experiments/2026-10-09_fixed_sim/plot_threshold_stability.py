"""Amendment 8's figure: ROC on a log false-alarm axis for step A's probe (level 1), with the cross-fit 1% and 5% points.

  uv run --no-project --python 3.13 --with matplotlib python experiments/2026-10-09_fixed_sim/plot_threshold_stability.py results/threshold_stability_<stamp>.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = Path(sys.argv[1])
rows = {r["id"]: r for r in json.loads(src.read_text())["rows"]}
CURVES = [("step A|level1|fixed_vs_e1|all 10", "fixed sim vs real, all 10 environments", "#1f6fb4", "-"),
          ("step A|level1|fixed_vs_e1|test 5", "fixed sim vs real, 5 test environments", "#7fb0dc", "-"),
          ("step A|level1|e2_vs_e1|all 10", "real vs real (control), all 10", "#888888", "--")]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for key, label, color, ls in CURVES:
    r = rows[key]
    fpr, tpr = r["roc"]["fpr"], r["roc"]["tpr"]
    pts = [(f, t) for f, t in zip(fpr, tpr) if f > 0]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, ls=ls, lw=2, label=f"{label} (AUROC {r['auroc']:.2f})")
    for pct in (1, 5):
        x = r[f"at_{pct}pct"]
        if key.endswith("all 10") and "fixed" in key:
            ax.errorbar(x["fpr"], x["caught_share"], yerr=[[x["caught_share"] - x["caught_ci95"][0]], [x["caught_ci95"][1] - x["caught_share"]]],
                        fmt="o", color=color, capsize=3, ms=6, zorder=5)
            ax.annotate(f"{100 * x['caught_share']:.0f}% caught at the {pct}% target\n(cross-fit, 95% interval)", (x["fpr"], x["caught_share"]),
                        textcoords="offset points", xytext=(-150, 40) if pct == 1 else (-160, 45), fontsize=8, color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.8), bbox=dict(fc="white", ec="none", alpha=0.9))
grid = [10 ** (-3 + 3 * i / 200) for i in range(201)]
ax.plot(grid, grid, color="#cccccc", lw=1, zorder=0, label="chance")
ax.set_xscale("log")
ax.set_xlim(1e-3, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("false alarms on real runs (log scale; left is better)")
ax.set_ylabel("simulated runs caught (higher is better)")
ax.set_title("Step A's probe, level 1: catch vs false alarms")
ax.legend(fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()
out = src.with_suffix(".png")
fig.savefig(out, dpi=150)
print("->", out)
