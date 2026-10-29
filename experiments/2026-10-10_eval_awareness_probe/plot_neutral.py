"""Plot the neutral-line controls: eval line minus the Linux line, per readout and placement, as % of each readout's reference.

  uv run --no-project --with matplotlib python experiments/2026-10-10_eval_awareness_probe/plot_neutral.py --b results/neutral_analysis_<B>.json --a results/neutral_analysis_A_per_token_<stamp>.json --old results/neutral_analysis_old_per_token_<stamp>.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
for x in ("b", "a", "old"):
    ap.add_argument(f"--{x}", required=True)
a = ap.parse_args()
res = {name: json.loads((HERE / f).read_text()) for name, f in (("Turn start (B)", a.b), ("Content, retrained (A)", a.a), ("Content, original probe", a.old))}
K = "specific effect (eval minus linux)"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, place, title in zip(axes, ("far", "near"), ("Line at the end of the system prompt", "Line just before each calling turn")):
    names = list(res)
    m = [100 * res[n][place][K]["fraction"] for n in names]
    lo = [100 * res[n][place][K]["fraction 95%"][0] for n in names]
    hi = [100 * res[n][place][K]["fraction 95%"][1] for n in names]
    bars = ax.bar(range(3), m, yerr=[[v - l for v, l in zip(m, lo)], [h - v for v, h in zip(m, hi)]], capsize=4,
                  color=["#4C72B0", "#9AA5B1", "#9AA5B1"])
    ax.bar_label(bars, labels=[f"{v:+.1f}" for v in m], padding=3, fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(3), names, fontsize=9)
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("eval line minus neutral line,\n% of the probe's synthetic size (higher = larger eval-minus-Linux effect)")
fig.suptitle("Far from the turn, only the turn-start readout moves more for the eval line than for a neutral line", fontsize=11)
fig.text(0.5, -0.03, "99 real honest LinuxArena runs. Neutral line: \"This is a Linux machine.\" 95% intervals: 2,000 draws of LinuxArena and synthetic tasks.",
         ha="center", fontsize=8)
fig.tight_layout()
out = HERE / Path(a.b).with_name("neutral_controls.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"-> {out}")
