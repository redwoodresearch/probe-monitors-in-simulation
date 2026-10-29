"""B-dev's figure: the frozen L63 probe's AUROC before and after evening the four slips (dev environments), fixed sim vs
real e1 beside the real-vs-real control, with 95% task-bootstrap intervals; the gap share removed and the planted
shift go in the caption text.

  uv run --no-project --python 3.13 --with matplotlib python experiments/2026-10-11_claimB_slips_evened_probe/plot_slips_evened.py results/slips_evened_<stamp>.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = Path(sys.argv[1])
r = json.loads(src.read_text())
b = r["L63"]
groups = [("fixed sim vs real", b["auroc_before"], b["auroc_after"]), ("real vs real (control)", b["auroc_e2_vs_e1_before"], b["auroc_e2_vs_e1_after"])]
fig, ax = plt.subplots(figsize=(5.6, 4.2))
for i, (label, before, after) in enumerate(groups):
    for j, (v, name, color) in enumerate(((before, "before", "#9bbbd9"), (after, "after evening", "#1f6fb4"))):
        x = i * 2.6 + j
        err = [[v["value"] - v["ci95"][0]], [v["ci95"][1] - v["value"]]]
        ax.bar(x, v["value"], color=color, width=0.9, label=name if i == 0 else None)
        ax.errorbar(x, v["value"], yerr=err, color="black", capsize=4)
        ax.text(x, v["ci95"][1] + 0.01, f"{v['value']:.3f}", ha="center", fontsize=10)
ax.axhline(0.5, color="#888888", lw=1, ls="--")
ax.set_xticks([0.5, 3.1], [g[0] for g in groups])
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("AUROC, frozen L63 probe (higher = more separable)")
ax.set_title("Evening the four leftover slips, dev environments")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
out = src.with_suffix(".png")
fig.savefig(out, dpi=150)
print("->", out)
