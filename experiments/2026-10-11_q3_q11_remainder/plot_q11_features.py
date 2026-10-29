"""Figure for q11_features_POSTHOC_*.json: share of the bash per-output score gap (sim − real e1) explained by each
structural feature (Gelbach), with intervals, plus all features together, the held-out best formatting set (the bar)
and the matching re-measure. Reads only the committed result file.

  uv run --no-project --with matplotlib python experiments/2026-10-11_q3_q11_remainder/plot_q11_features.py <result json>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
path = Path(sys.argv[1])
R = json.loads(path.read_text())
g = R["gelbach_sim_vs_real"]
feats = sorted(g["features"].items(), key=lambda kv: -abs(kv[1]["share"]))[:8]
rows = [(f"all features ({len(g['features'])})", 100 * g["share_explained_all"], [100 * x for x in g["share_explained_all_ci95"]], "#555555")]
hb = R["bar_heldout"]
rows.append((f"best ≤3 formatting, held out\n({', '.join(hb['chosen_on_selection_half'])})", 100 * hb["share_on_evaluation_half"],
             [100 * x for x in hb["share_on_evaluation_half_ci95"]], "#2a77d4"))
m = R["matching"]
rows.append((f"matching re-measure\n(coverage {100 * m['coverage_sim_weight']:.0f}%)", 100 * m["share_explained"],
             [100 * x for x in m["share_explained_ci95"]], "#888888"))
for k, v in feats:
    rows.append((f"{k} ({v['class']})", 100 * v["share"], [100 * x for x in v["share_ci95"]], "#2a77d4" if v["class"] == "formatting" else "#e8693a"))
fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(rows) + 1.6))
for i, (lab, v, (lo, hi), col) in enumerate(rows):
    yy = len(rows) - 1 - i
    ax.barh(yy, v, color=col, height=0.65)
    ax.errorbar(v, yy, xerr=[[v - lo], [hi - v]], color="k", capsize=3, lw=1)
    ax.text(max(hi, v, 0) + 2, yy, f"{v:.0f}%", va="center", fontsize=9)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=9)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(50, color="#2a77d4", lw=1, ls="--")
ax.text(50, len(rows) - 0.4, " bar: 50%", color="#2a77d4", fontsize=8, va="bottom")
ax.set_xlabel("share of the bash per-output score gap explained (%)  (sim − real e1, frozen L63 probe)")
ax.set_title(f"Bash replies: {R['reading']}", loc="left", fontsize=11)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = HERE / "figures" / f"{path.stem}.png"
fig.savefig(out, dpi=150)
print(out)
