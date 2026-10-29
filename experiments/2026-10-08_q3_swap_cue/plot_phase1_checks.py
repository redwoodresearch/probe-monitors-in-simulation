"""Figure for analyze_phase1_checks.py: catch at 1% false alarms for real e1, real e2 and simulated attacks, on the 217
selected cells and on all 939 cells, with 95% intervals (thresholds redrawn per draw) and the planted checks' marks.
The 217-cell null is labelled assumption-dependent (its cells were chosen on simulated-run features), as the result
file states; the 939-cell null is the selection-free one (review on #581, 2026-10-11). Each run writes a new
timestamped PNG, so earlier figures are kept.

  uv run --no-project --with numpy --with matplotlib python plot_phase1_checks.py results/phase1_checks_POSTHOC_<stamp>.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).parent
REAL, SIM, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
RATE, MAIN = "0.01", "output_end"


def main(path):
    d = json.loads(Path(path).read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, facecolor=SURFACE)
    sets = (("217_selected_cells", "217 selected cells", "\n(assumption-dependent: cells chosen on sim features)"), ("939_all_cells", "all 939 cells", ""))
    for ax, (name, title, qualifier) in zip(axes, sets):
        st = d["sets"][name]
        r = st["readouts"][MAIN][RATE]
        p = st["null_primary_exact_sim_vs_e2"][RATE]["selection_aware_p"]
        g = r["gap_sim_minus_e2 (primary)"]
        arms = [("real e1 (trained on)", "e1", REAL), ("real e2", "e2", REAL), ("simulated", "sim", SIM)]
        for x, (label, key, color) in enumerate(arms):
            v, (lo, hi) = r["catch"][key], r["catch_ci95_thresholds_redrawn"][key]
            ax.bar(x, v, width=0.6, color=color, edgecolor=SURFACE, linewidth=2)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=4, linewidth=1.2)
            ax.text(x, hi + 0.015, f"{100 * v:.0f}%", ha="center", va="bottom", color=INK, fontsize=12, fontweight="bold")
        marks = []
        for s in ("+10", "-10"):
            pl = r["plants"][s]
            sign = s.replace("-", "−")
            marks.append(f"{sign} not constructible" if "on sim - e2 (primary)" not in pl
                         else f"{sign} {'detected' if pl['on sim - e2 (primary)']['detected'] else 'missed'}")
        f = lambda x: f"{100 * x:+.1f}".replace("-", "−")
        ax.set_title(f"{title}\nsim − e2: {f(g['observed'])} pts [{f(g['ci95_thresholds_redrawn'][0])}, {f(g['ci95_thresholds_redrawn'][1])}],"
                     f" null p {p:.4f}{qualifier}\nplanted ±10: {', '.join(marks)}", color=INK, fontsize=10)
        ax.set_xticks(range(3), [a[0] for a in arms], color=INK)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("flagged, targeting 1% false alarms (higher = more)", color=INK)
    axes[0].set_ylim(0, 0.62)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "figures" / f"{Path(path).stem}_qualified_{stamp}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
