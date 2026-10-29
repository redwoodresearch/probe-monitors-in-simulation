"""Figure for analyze_fix939.py, all 939 cells: as-is sim − e2, fixed sim − e2 and the e2 − e1 control, in points with
95% intervals (thresholds redrawn per draw), per readout, at 5% and 1% false alarms, with the ±5-point band and the
fixed arm's planted checks. The content-only group is labelled with its own cohort (runs and tasks, read from the
result), which is smaller than the 939 cells (review on #603, 2026-10-11). Each run writes a new timestamped PNG, so
earlier figures are kept.

  uv run --no-project --with numpy --with matplotlib python plot_fix939.py results/fix939_POSTHOC_<stamp>.json

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
REAL, ASIS, FIXED, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
GROUPS = [("output_end", "simulator replies"), ("call_end", "agent commands"), ("call_end_content", "agent commands,\ncontent only")]
BARS = [("as_is_sim_minus_e2", "sim − e2, as simulated", ASIS), ("fixed_sim_minus_e2 (primary)", "sim − e2, format fixed", FIXED),
        ("real_e2_minus_e1 (e1 was in training)", "e2 − e1 (e1 trained on)", REAL)]


def mark(pl):
    s = {"+10": "+10", "-10": "−10"}
    return " ".join(f"{s[k]} n/c" if "needed" in v else f"{s[k]} {'✓' if v['detected'] else '✗'}" for k, v in sorted(pl.items()))


def pts(x):
    return f"{100 * x:+.1f}".replace("-", "−")


def main(path):
    d = json.loads(Path(path).read_text())["readouts"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, facecolor=SURFACE)
    width = 0.26
    for ax, rate in zip(axes, ("0.05", "0.01")):
        ax.axhspan(-5, 5, color="#efeeea", zorder=0)
        for gi, (ro, label) in enumerate(GROUPS):
            r = d[ro][rate]
            for bi, (key, name, color) in enumerate(BARS):
                v, (lo, hi) = r[key]["observed"], r[key]["ci95"]
                x = gi + (bi - 1) * width
                ax.bar(x, 100 * v, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=name if gi == 0 else None, zorder=2)
                ax.errorbar(x, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], color=INK, capsize=3, linewidth=1.1, zorder=3)
                bottom = 100 * min(lo, v, 0)  # the lowest point of the bar and its whisker
                # one row of values just under the -5 line, or under the whisker where it reaches below; clear of the band edges
                y = -6.0 if bottom > -5.5 else bottom - 1.0
                ax.text(x, y, pts(v), ha="center", va="top", color=INK, fontsize=9, fontweight="bold",
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6}, zorder=4)
            ax.text(gi, 7.0, f"fixed-arm plants\n{mark(r['plants'])}", ha="center", va="bottom", color=MUTED, fontsize=8)
        ax.axhline(0, color=INK, linewidth=0.9, zorder=2)
        labels = [g[1] + (f" ({d[g[0]]['cells']} runs,\n{d[g[0]]['tasks']} tasks)" if g[0] == "call_end_content" else "") for g in GROUPS]
        ax.set_xticks(range(len(GROUPS)), labels, color=INK)
        b = d["output_end"][rate]["bar"]
        ax.set_title(f"targeting {float(rate):.0%} false alarms · replies: {b['reading']}", color=INK, fontsize=11)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.set_axisbelow(True)
    r5 = d["output_end"]["0.05"]
    axes[0].set_ylabel("difference in attacks flagged (points); band = ±5", color=INK)
    axes[0].set_ylim(-21, 10)
    axes[0].legend(frameon=False, loc="lower right", labelcolor=INK, fontsize=9)
    fig.suptitle(f"Fixing the simulator's output format: replies gap {pts(r5['as_is_sim_minus_e2']['observed'])} → "
                 f"{pts(r5['fixed_sim_minus_e2 (primary)']['observed'])} pts at 5% (all 939 cells)", color=INK, fontsize=12)
    fig.tight_layout()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "figures" / f"{Path(path).stem}_qualified_{stamp}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
