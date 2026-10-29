"""Late-run false alarms figure (post hoc): per row, real − sim and real − run 2 false-alarm rate on read-only calls at the
frozen 1% threshold, with pointwise 95% intervals (thresholds re-estimated; draw 0's models held fixed) and the ±1 band.

  uv run --no-project --with matplotlib python plot_POSTHOC_late_run_false_alarms.py <late_run_false_alarms_POSTHOC_*.json>
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
    segs = list(R["segments"].items())
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for y, (name, s) in enumerate(segs[::-1]):
        for dy, key, color, lab in ((0.15, "real_minus_sim", "#4C72B0", "real − simulated"), (-0.15, "real_minus_run2", "gray", "real run 1 − run 2 (yardstick)")):
            v = s[key]
            if v["point"] is None:
                continue
            if v["ci95"]:  # a percentile interval need not contain its point, so it is drawn on its own
                ax.hlines(y + dy, 100 * v["ci95"][0], 100 * v["ci95"][1], color=color, lw=2, label=lab if y == 0 else None)
            ax.plot(100 * v["point"], y + dy, "o", color=color, ms=6)
            txt = f"{100 * v['point']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]" if v["ci95"] else f"{100 * v['point']:+.1f} (not reported)"
            ax.annotate(txt, (100 * v["point"], y + dy), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
        n, r = s["calls"], s["readings"]
        ax.text(1.01, y, f"difference: {r['difference_from_zero']}; ±1: {r['equivalence_1pt'].split(' (')[0]}\nread-only calls: real {n['real']['read_only']}, sim {n['sim']['read_only']}, run 2 {n['run2']['read_only']}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7)
    ax.axvspan(-1, 1, color="#55A868", alpha=0.08, label="±1-point margin")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(segs)))
    ax.set_yticklabels([n for n, _ in segs[::-1]], fontsize=9)
    ax.set_xlabel("difference in read-only commands flagged (false alarms) at the frozen 1% threshold (percentage points; 0 = agree)\n"
                  "pointwise 95% intervals, thresholds re-estimated, draw 0's models held fixed")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("Late-run false alarms (post hoc; point rates seen before the rule): real vs simulated read-only calls", fontsize=10)
    fig.tight_layout()
    dst = HERE / "figures" / f"{Path(path).stem}.png"
    dst.parent.mkdir(exist_ok=True)
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    main(ap.parse_args().path)
