"""Figure for q11_remainder_POSTHOC_*.json: the frozen L63 probe's remaining gap (fixed sim − real e1, after B-dev's
evening) split by reply type, with the real e2 − e1 control, and by fix-touched vs untouched. Reads only the result file.

  uv run --no-project --with matplotlib python experiments/2026-10-11_q3_q11_remainder/plot_q11_remainder.py <result json>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SIMC, CTRL = "#e8693a", "#2a77d4"
src = Path(sys.argv[1])
d = json.loads(src.read_text())
a, e = d["after"]["by_kind"], d["control_e2_minus_e1_after"]["by_kind"]
kinds = ["shell", "str_replace", "create", "insert", "view", "editor_other"]
names = {"shell": "shell (bash)", "str_replace": "str_replace", "create": "create", "insert": "insert", "view": "view",
         "editor_other": "other editor"}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
for i, k in enumerate(kinds):
    for off, blk, col in ((-0.19, a, SIMC), (0.19, e, CTRL)):
        v = blk["categories"][k]
        x, (lo, hi) = v["contribution"], v["contribution_ci95"]
        ax.barh(i + off, x, 0.36, color=col)
        ax.errorbar(x, i + off, xerr=[[x - lo], [hi - x]], color="k", capsize=3, lw=1)
    v = a["categories"][k]
    ax.text(max(v["contribution_ci95"][1], 0) + 0.006, i - 0.19, f"{100 * v['share']:.0f}% of gap", va="center", fontsize=9)
ax.set_yticks(range(len(kinds)), [names[k] for k in kinds])
ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("contribution to the run-score gap (logits; right = more sim-like)")
ax.set_title(f"By reply type: sim − real e1 gap {a['gap']:.3f}; real e2 − e1 {e['gap']:+.3f}", loc="left", fontsize=10)
from matplotlib.patches import Patch

ax.legend(handles=[Patch(color=SIMC, label="fixed sim − real e1"), Patch(color=CTRL, label="real e2 − real e1 (control)")],
          frameon=False, loc="lower right", fontsize=9)
groups = [("touched by a slip fix", d["after"]["by_touched"]["categories"]["touched"]),
          ("untouched", d["after"]["by_touched"]["categories"]["untouched"]),
          ("str_replace + insert", d["after"]["by_family"]["categories"]["str_replace+insert"]),
          ("all other replies", d["after"]["by_family"]["categories"]["rest"])]
for i, (lab, v) in enumerate(groups):
    x, (lo, hi) = 100 * v["share"], (100 * q for q in v["share_ci95"])
    bx.barh(i, x, 0.6, color=SIMC)
    bx.errorbar(x, i, xerr=[[x - lo], [hi - x]], color="k", capsize=3, lw=1)
    bx.text(max(hi, 0) + 3, i, f"{x:.0f}%", va="center", fontsize=9)
bx.set_yticks(range(len(groups)), [g[0] for g in groups])
bx.invert_yaxis()
bx.axvline(0, color="k", lw=0.8)
bx.axhline(1.5, color="#bbbbbb", lw=0.8)
bx.set_xlim(-30, 140)
bx.set_xlabel("share of the sim − real e1 gap (%)")
bx.set_title("Fix-touched vs not; editor family", loc="left", fontsize=10)
p = d["planted"]["bar"]
fig.suptitle(f"The probe's remaining gap sits in untouched shell replies (planted check: {100 * p['share']:.0f}% of an "
             f"R2-undo shift lands in str_replace, {'pass' if p['passed'] else 'fail'})", fontsize=11)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
    bx.spines[s_].set_visible(False)
fig.tight_layout()
out = HERE / "figures" / src.name.replace(".json", ".png")
fig.savefig(out, dpi=150)
print(out)
