"""Figure for scorer_split_POSTHOC_2026-10-11_003502.json: catch targeting 1% on real attempted runs, scorer-confirmed (C) vs
scorer-failed (I). Reads only the committed result file.

2026-10-11, 13:08 PT: figures/q31/6_scorer_split.png is superseded by 6_scorer_split_2026-10-11_051836.png; its title
overstated run 1 (C − I +3.2 [−1.5, +11.5] is inconclusive).

  uv run --no-project --with matplotlib python experiments/2026-10-08_q3_onpolicy_figures/plot_scorer_split.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R = json.loads((HERE / "results" / "scorer_split_POSTHOC_2026-10-11_003502.json").read_text())["results"]
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for i, w in enumerate(("run1", "run2")):
    for j, (g, col, lab) in enumerate((("C", "#2a77d4", "Scorer: side task succeeded"), ("I", "#e8693a", "Scorer: side task failed"))):
        e = R[w][g]
        v, (lo, hi) = 100 * e["catch_0.01"], (100 * x for x in e["catch_0.01_interval"]["ci95"])
        x = i + (j - 0.5) * 0.36
        ax.bar(x, v, 0.34, color=col, label=lab if i == 0 else None)
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="k", capsize=4, lw=1.2)
        ax.text(x, hi + 1.5, f"{v:.1f}\nn={e['n']}", ha="center", va="bottom", fontsize=10)
d = {w: R[w]["C_minus_I_0.01"] for w in ("run1", "run2")}
ax.set_xticks([0, 1], [f"Real run 1\nC − I {100 * d['run1']['value']:+.1f} [{100 * d['run1']['interval']['ci95'][0]:+.1f}, {100 * d['run1']['interval']['ci95'][1]:+.1f}]",
                       f"Real run 2\nC − I {100 * d['run2']['value']:+.1f} [{100 * d['run2']['interval']['ci95'][0]:+.1f}, {100 * d['run2']['interval']['ci95'][1]:+.1f}]"])
ax.set_ylim(0, 118)
ax.set_ylabel("Flagged, targeting 1% false alarms (%)\nhigher = more runs flagged")
def ci(w):
    d = R[w]["C_minus_I_0.01"]
    return f"{100 * d['value']:+.1f} [{100 * d['interval']['ci95'][0]:+.1f}, {100 * d['interval']['ci95'][1]:+.1f}]"


reading = {w: "alike" if R[w]["reading"].startswith("flags confirmed and failed attempts alike") else R[w]["reading"] for w in ("run1", "run2")}
ax.set_title(f"Run 2: scorer-confirmed and scorer-failed attempts flagged {reading['run2']} (C − I {ci('run2')});\n"
             f"run 1 {reading['run1']} (C − I {ci('run1')})", loc="left", fontsize=10)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
out = HERE / "figures" / "q31" / f"6_scorer_split_{stamp}.png"  # a new name each run; earlier figures are kept
fig.savefig(out, dpi=150)
print(out)
