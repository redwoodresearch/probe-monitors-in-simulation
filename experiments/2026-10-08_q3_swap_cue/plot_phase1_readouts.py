"""Figure for analyze_phase1_checks.py, all 939 cells: the sim − e2 gap and the e2 − e1 control, in points with 95%
intervals (thresholds redrawn per draw), per readout (simulator replies, agent commands, agent commands content-only),
at 5% and 1% false alarms, with the planted checks' marks. The commands side reads 'no difference detected': its planted
check was not constructible at 5% (commands) and 1% (content only), so sensitivity is untested there (review on #598,
2026-10-11). Each run writes a new timestamped PNG, so earlier figures are kept.

  uv run --no-project --with numpy --with matplotlib python plot_phase1_readouts.py results/phase1_checks_POSTHOC_<stamp>.json

Layout fix (2026-10-11, about 10:20 PT, from session 09's figure audit): value labels now sit in one row just under
the -5 line (or under a whisker that reaches below it), clear of the band edges and whiskers. Earlier renders of this
figure are superseded by the one this version writes; the numbers are unchanged.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REAL, SIM, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
GROUPS = [("output_end", "simulator replies"), ("call_end", "agent commands"), ("call_end_content", "agent commands,\ncontent only")]
BARS = [("gap_sim_minus_e2 (primary)", "sim − e2", SIM), ("real_e2_minus_e1 (e1 was in training)", "e2 − e1 (e1 trained on)", REAL)]


def mark(pl):
    s = {"+10": "+10", "-10": "−10"}
    return " ".join(f"{s[k]} n/c" if "needed" in v else f"{s[k]} {'✓' if v['on sim - e2 (primary)']['detected'] else '✗'}"
                    for k, v in sorted(pl.items()))


def main(path):
    d = json.loads(Path(path).read_text())["sets"]["939_all_cells"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True, facecolor=SURFACE)
    width = 0.36
    for ax, rate in zip(axes, ("0.05", "0.01")):
        for gi, (ro, label) in enumerate(GROUPS):
            r = d["readouts"][ro][rate]
            for bi, (key, name, color) in enumerate(BARS):
                v, (lo, hi) = r[key]["observed"], r[key]["ci95_thresholds_redrawn"]
                x = gi + (bi - 0.5) * width
                ax.bar(x, 100 * v, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=name if gi == 0 else None)
                ax.errorbar(x, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], color=INK, capsize=3, linewidth=1.1)
                bottom = 100 * min(lo, v, 0)  # the lowest point of the bar and its whisker
                # one row of values just under the -5 line, or under the whisker where it reaches below; clear of the band edges
                y = -6.0 if bottom > -5.5 else bottom - 1.0
                ax.text(x, y, f"{100 * v:+.1f}".replace("-", "−"), ha="center", va="top", color=INK, fontsize=10, fontweight="bold",
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6}, zorder=4)
            ax.text(gi, 7.2, f"plants {mark(r['plants'])}", ha="center", va="bottom", color=MUTED, fontsize=8.5)
        ax.axhline(0, color=INK, linewidth=0.9)
        ax.set_xticks(range(len(GROUPS)), [g[1] for g in GROUPS], color=INK)
        ax.set_title(f"targeting {float(rate):.0%} false alarms", color=INK, fontsize=11)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("difference in attacks flagged (points)", color=INK)
    axes[0].set_ylim(-21, 9)
    axes[0].legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.suptitle("Reading replies: 10–14 pts fewer sim attacks caught; reading commands: no difference detected\n"
                 "(planted check not constructible at 5% commands and 1% content, so sensitivity untested there)", color=INK, fontsize=12)
    fig.tight_layout()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "figures" / f"phase1_readouts_POSTHOC_{Path(path).stem.split('POSTHOC_')[1]}_qualified_{stamp}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
