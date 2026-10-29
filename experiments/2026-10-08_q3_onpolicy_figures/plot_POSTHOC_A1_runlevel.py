"""A1 figure (post hoc): A - B in the share of honest runs with any alarm in their first 20 read-only calls, beside the
attack run 1 - run 2 control and the planted check, with 95% intervals (thresholds re-estimated) and the +-10 margin. $0.

  uv run --no-project --with matplotlib python plot_POSTHOC_A1_runlevel.py <A1_runlevel_POSTHOC_*.json>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent


def main(path: str):
    R = json.loads(Path(path).read_text())
    ab, rr, pl = R["A_minus_B"]["first 20"], R["run1_minus_run2"]["first 20"], R["planted"]
    rows = [(f"honest A − B\n({100 * ab['share']['honA']:.1f}% vs {100 * ab['share']['honB']:.1f}% of 190 runs)", ab["difference"], ab["ci95_thresholds_reestimated"], "#4C72B0"),
            (f"attack run 1 − run 2 (control)\n({100 * rr['share']['real']:.1f}% vs {100 * rr['share']['run2']:.1f}% of {R['run1_minus_run2']['tasks']} runs)", rr["difference"], rr["ci95_thresholds_reestimated"], "gray"),
            (f"A − B with {len(pl['planted'])} planted B alarms\n(check; {'detected' if pl['detected'] else 'not detected'})", pl["difference"], pl["ci95"], "#C44E52")]
    rows = rows if pl["planted"] else rows[:2]  # no plant was made: the check is skipped, not drawn
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for y, (lab, d, iv, c) in enumerate(rows[::-1]):  # a percentile interval need not contain its point, so it is drawn on its own
        if iv:
            ax.hlines(y, 100 * iv[0], 100 * iv[1], color=c, lw=2)
            ax.vlines([100 * iv[0], 100 * iv[1]], y - 0.12, y + 0.12, color=c, lw=2)
        ax.plot(100 * d, y, "o", color=c, ms=8)
        txt = f"{100 * d:+.1f} [{100 * iv[0]:+.1f}, {100 * iv[1]:+.1f}]" if iv else f"{100 * d:+.1f} (interval not reported)"
        ax.annotate(txt, (100 * d, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10)
    ax.axvspan(-10, 10, color="#55A868", alpha=0.08, label="±10-point equivalence margin")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9)
    ax.set_xlabel("difference in runs with any alarm in the first 20 read-only calls\n(percentage points; 95% intervals, thresholds re-estimated; 0 = agree)")
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.legend(fontsize=8, loc="lower right")
    r = R["readings_primary"]
    ax.set_title(f"A1 (post hoc), honest A − B:\ndifference from zero {r['difference_from_zero']}; within ±10 points: {r['equivalence_10']}", fontsize=10)
    fig.tight_layout()
    dst = HERE / "figures" / f"{Path(path).stem}_v2.png"  # v2: the label-fixed figure; the unsuffixed v1 is superseded (s3_manifest_A1_runlevel_v2.json)
    dst.parent.mkdir(exist_ok=True)
    fig.savefig(dst, dpi=150)
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    main(ap.parse_args().path)
