"""Per-task false alarms figure (exploratory): per world, the pooled per-call false-alarm rate beside the per-task mean
(each task once), at the frozen 1% threshold; the paired differences with intervals in the title. $0.

  uv run --no-project --with matplotlib python plot_POSTHOC_per_task_false_alarms.py <per_task_false_alarms_POSTHOC_*.json>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
NAME = {"real": "real run 1", "sim": "simulated", "run2": "real run 2"}


def main(path: str):
    R = json.loads(Path(path).read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = list(NAME)
    for j, (key, lab, color) in enumerate((("pooled", "pooled (per call)", "#BBBBBB"), ("per_task_mean", "each task once", "#4C72B0"))):
        vals = [100 * R["worlds"][w][key] for w in xs]
        bars = ax.bar([i + (j - 0.5) * 0.38 for i in range(len(xs))], vals, 0.38, color=color, label=lab)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}%", (b.get_x() + b.get_width() / 2, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([NAME[w] for w in xs])
    ax.set_ylabel("read-only calls flagged at the frozen 1% threshold (%)\n(lower is better)")
    c = R["comparisons"]
    f = lambda d: f"{100 * d['point']:+.2f} [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}]"
    ax.set_title("Exploratory; 95% intervals in percentage points, thresholds re-estimated, draw 0's models fixed\n"
                 f"Real − sim: each task once {f(c['real - sim']['per_task'])}; pooled {f(c['real - sim']['pooled'])}\n"
                 f"Run 1 − run 2: each task once {f(c['real - run2']['per_task'])}; pooled {f(c['real - run2']['pooled'])}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    dst = HERE / "figures" / f"{Path(path).stem}_v2.png"  # v2: the exploratory label and interval note in the title; v1 is superseded (s3_manifest_per_task_false_alarms_v2.json)
    dst.parent.mkdir(exist_ok=True)
    fig.savefig(dst, dpi=150)
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    main(ap.parse_args().path)
