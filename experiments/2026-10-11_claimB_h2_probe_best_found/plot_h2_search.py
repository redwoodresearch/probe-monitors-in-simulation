"""H2 figure (written before any H2 result): does the best probe found separate the fixed simulator from real runs more
than step A's probe? Left: test AUROC of step A's pipeline (A0) and the best found, fixed sim vs e1 and e2 vs e1 (the
real-vs-real baseline), each with its shuffle null's 95th percentile. Right: D = best found - A0 (fixed sim vs e1) with
its 95% interval over main-task draws and the 0.03 margin. Reads one h2_summary_*.json; numbers only.

  uv run --no-project --python 3.12 --with matplotlib python experiments/2026-10-11_claimB_h2_probe_best_found/plot_h2_search.py results/h2_summary_<stamp>.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
src = HERE / sys.argv[1]
s = json.loads(src.read_text())
if "fixed_vs_e1" not in s.get("rows", {}):
    raise SystemExit(f"nothing to plot: {s['decision']}")
rows, d = s["rows"], s["rows"]["fixed_vs_e1"]["D"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
groups = [("fixed_vs_e1", "fixed sim vs real e1"), ("e2_vs_e1", "real e2 vs real e1")]
for g, (comp, label) in enumerate(groups):
    for j, (key, null, name, color) in enumerate([("A0_auroc", "A0_null_p95", "step A's probe", "#4c72b0"),
                                                  ("best_found_auroc", "best_null_p95", "best found", "#dd8452")]):
        x, v = g + (j - 0.5) * 0.36, rows[comp][key]
        ax.bar(x, v - 0.5, 0.34, bottom=0.5, color=color, label=name if g == 0 else None)  # bars start at chance
        ax.text(x, 0.43, f"{v:.3f}", ha="center", fontsize=11, weight="bold", color=color)
        ax.plot([x - 0.17, x + 0.17], [rows[comp][null]] * 2, color="k", ls="--", lw=1, label="shuffle null p95" if g == j == 0 else None)
ax.set_xticks(range(len(groups)), [lab for _, lab in groups])
ax.set_ylim(0.4, 1.0)
ax.axhline(0.5, color="k", lw=0.6)
ax.set_ylabel("test AUROC (higher = more separable)")
ax.legend(frameon=False, fontsize=9, loc="upper right")
bx.vlines(0, *d["ci95"], color="#dd8452", lw=2)  # percentile interval, drawn by its ends (it need not contain the point)
bx.plot([-0.05, 0.05, None, -0.05, 0.05], [d["ci95"][0]] * 2 + [None] + [d["ci95"][1]] * 2, color="#dd8452", lw=2)
bx.plot([0], [d["value"]], "o", color="#dd8452")
bx.text(0.08, d["value"], f"{d['value']:+.3f}\n[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", va="center", fontsize=10)
bx.axhline(0.03, color="grey", ls=":", lw=1)  # the decision margin (one-sided: only +0.03 is used)
bx.axhline(0, color="k", lw=0.6)
bx.set_xlim(-0.3, 0.6)
bx.set_xticks([])
bx.set_ylabel("D = best found − step A (AUROC)")
bx.set_title(s["decision"], fontsize=10)
fig.tight_layout()
dst = src.with_suffix(".png")
fig.savefig(dst, dpi=150)
print("->", dst)
