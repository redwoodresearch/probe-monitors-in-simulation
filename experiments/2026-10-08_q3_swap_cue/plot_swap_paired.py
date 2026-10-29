"""One figure for James from analyze_swap_paired.py's output: the tool-output detector's catch rate on the same 100
simulated attack runs as simulated, with each formatting fix, with both, and on the real attacks of the same tasks, at
the 5% and 1% false-alarm thresholds. Error bars are paired: each bar's 95% interval of its difference from real.

  python plot_swap_paired.py --paired results/swap_paired_<stamp>.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
BARS = [("V1", "As simulated", "#9ca3af"), ("V4", "Insert fix", "#93c5fd"), ("SR", "Replace fix", "#93c5fd"),
        ("COMB", "Both fixes", "#1d4ed8"), ("real", "Real runs", "#15803d")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", required=True)
    ap.add_argument("--subset", help="for a pooled analysis file: which set of runs (default: pooled)")
    args = ap.parse_args()
    src = Path(args.paired)
    res = json.loads(src.read_text())["readouts"]["output_end"]
    if "runs" not in res:  # a multi-set analysis file holds one entry per set of runs
        sets = [k for k in res if k != "real_honest_false_alarms"]
        name = args.subset or ("pooled" if "pooled" in sets else sets[0] if len(sets) == 1 else None)
        if name not in sets:
            raise SystemExit(f"pass --subset, one of {sets}")
        res = res[name]
    bars = [b for b in BARS if b[0] in res["0.05"]["arms"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for ax, (a, name) in zip(axes, (("0.05", "5%"), ("0.01", "1%"))):
        arms, diffs = res[a]["arms"], res[a]["differences"]
        ax.axhline(100 * arms["real"]["catch"], color="#15803d", ls="--", lw=1, zorder=0)
        for x, (lab, tick, color) in enumerate(bars):
            v = 100 * arms[lab]["catch"]
            ax.bar(x, v, color=color, width=0.7)
            if lab != "real":
                dd = diffs[f"{lab} - real"]
                lo, hi = 100 * (dd["ci95"][0] - dd["observed"]), 100 * (dd["ci95"][1] - dd["observed"])
                ax.errorbar(x, v, yerr=[[-lo], [hi]], color="black", capsize=4, lw=1.2)
            else:
                for real, dx in (("real-attack-e1", -0.12), ("real-attack-e2", 0.12)):
                    ax.plot(x + dx, 100 * arms[real]["catch"], "o", color="black", ms=4)
            ax.text(x, v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(bars)), [t for _, t, _ in bars])
        ax.set_title(f"Threshold set for {name} false alarms on real honest runs", fontsize=10)
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Attacks caught (%)\nhigher means more attacks caught")
    fig.suptitle("Do two formatting fixes make simulated attack catch rates match real ones?", fontsize=13)
    n = res["runs"]
    fig.text(0.5, 0.015, f"The detector reads tool outputs. The same {n} simulated attack runs ({res['tasks']} tasks): as simulated, "
             f"with {'file-insert or file-replace outputs' if len(bars) > 3 else 'file-insert and file-replace outputs'} shown the way real runs show them{', or both' if len(bars) > 3 else ''}.\nReal runs: the real attacks on "
             f"the same tasks (two runs averaged; dots show each; dashed line). Error bars: 95% interval of each bar's difference "
             f"from real,\nresampling tasks; a bar whose error bar crosses the dashed line is not distinguishable from real. "
             f"False alarms on the fixed runs are assumed, not measured.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    out = HERE / "figures" / f"{src.stem}{'_' + args.subset if args.subset else ''}.png"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    fig.savefig(out, dpi=150)
    print("->", out)


if __name__ == "__main__":
    main()
