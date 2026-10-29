"""The rendering swap as differences from real runs, one point and paired 95% interval per row, from a pooled
analyze_swap_paired.py output. $0, local. The bar version (plot_swap_paired.py) is kept as a backup.

Rows: as simulated, each single fix (the 100-run set, the only one that has them), and both fixes, each minus real (the
e1 / e2 average of the same cells); real e2 − e1 as the yardstick; and both fixes − as simulated, the effect of the fixes.

  python plot_swap_differences.py --paired results/swap_paired_<stamp>.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
FIRST = "swap_frozen_runs_2026-10-08_044006"  # the 100-run set with the single-fix arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", required=True)
    args = ap.parse_args()
    src = Path(args.paired)
    res = json.loads(src.read_text())["readouts"]["output_end"]
    pooled, first = res["pooled"], res[FIRST]
    n = lambda s: f"{s['runs']} runs, {s['tasks']} tasks"  # noqa: E731
    rows = [("As simulated − real", pooled, "V1 - real", "#6b7280"),
            ("Insert fix only − real", first, "V4 - real", "#60a5fa"),
            ("Replace fix only − real", first, "SR - real", "#60a5fa"),
            ("Both fixes − real", pooled, "COMB - real", "#1d4ed8"),
            ("Real rerun − real (yardstick: e2 − e1)", pooled, "real-attack-e2 - real-attack-e1", "#15803d"),
            (None, None, None, None),
            ("Both fixes − as simulated (the fixes' effect)", pooled, "COMB - V1", "#1d4ed8")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, (a, name) in zip(axes, (("0.05", "5%"), ("0.01", "1%"))):
        ax.axvspan(-5, 5, color="#e5e7eb", zorder=0)
        ax.text(5.4, -0.75, "shaded: ±5 pts", ha="left", va="center", fontsize=8.5, color="#4b5563")
        ax.axvline(0, color="black", lw=2, zorder=1)
        for y, (label, sub, key, color) in enumerate(rows):
            if label is None:
                ax.axhline(y, color="#9ca3af", lw=0.8, ls=":")
                continue
            d = sub[a]["differences"][key]
            v, lo, hi = (100 * x for x in (d["observed"], *d["ci95"]))
            ax.plot([lo, hi], [y, y], color=color, lw=2.5, solid_capstyle="round", zorder=2)
            ax.plot(v, y, "o", color=color, ms=8, zorder=3)
            ax.text(27, y, f"{v:+.1f}  [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=9)  # a fixed column, clear of the zero line
        ax.set_title(f"Threshold set for {name} false alarms on real honest runs", fontsize=10)
        ax.set_xlim(-32, 46)
        ax.set_xlabel("difference in attacks caught (percentage points)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(rows)), [f"{r[0]}\n({n(r[1])})" if r[0] else "" for r in rows], fontsize=9)
    axes[0].set_ylim(len(rows) - 0.4, -1.1)
    fig.suptitle("How far is each simulated version from real runs?", fontsize=13)
    fig.text(0.5, 0.015, "Each point is a paired difference with its 95% interval (tasks resampled): an interval that crosses "
             "zero means no detectable difference, and its two ends bound the differences the data still allow. Detector reading tool "
             "outputs; false alarms on the fixed runs are assumed, not measured.", ha="center", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = HERE / "figures" / f"swap_differences_{src.stem.removeprefix('swap_paired_')}.png"
    if out.exists():
        raise FileExistsError(out)
    fig.savefig(out, dpi=150)
    print("->", out)


if __name__ == "__main__":
    main()
