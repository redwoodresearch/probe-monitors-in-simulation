"""Catch rate on real attack (epoch 1, epoch 2) and simulated attack runs at a frozen 1% and 5% false-alarm rate on
real honest runs, per detector, with 95% main-task bootstrap intervals. Reads a metrics file from analyze_q3_attack.py.

  uv run --with matplotlib python experiments/2026-10-06_q3_attack_detector/plot_q3_attack.py --metrics results/metrics_<run_id>__<stamp>.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DETECTORS = [("probe_call_end_content_mean", "Probe,\ncontent-token\ncalls (headline)"),
             ("tfidf_call_end_content_mean", "TF-IDF,\ncontent-token\ncalls"),
             ("probe_call_end_timeout_mean", "Probe,\ntimeout-ending\ncalls"),
             ("tfidf_call_end_timeout_mean", "TF-IDF,\ntimeout-ending\ncalls"),
             ("probe_call_end_mean", "Probe,\nall calls"),
             ("probe_output_end_mean", "Probe,\ntool outputs"),
             ("timeout_share_and_tool_mix", "Timeout share\n+ tool mix"),
             ("length_only", "Length only")]
ARMS = [("e1", "real attack, epoch 1", "#2a78d6"), ("e2", "real attack, epoch 2", "#eb6834"),
        ("sim", "simulated attack", "#1baf7a")]  # the reference palette's first three slots (all-pairs validated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()
    m = json.loads((HERE / args.metrics).read_text() if not Path(args.metrics).is_absolute() else Path(args.metrics).read_text())
    detectors = [d for d in DETECTORS if d[0] in m["readouts"]]  # earlier runs lack the timeout-ending readouts
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharey=True)
    width = 0.26
    for ax, fpr in zip(axes, ("0.01", "0.05")):
        for j, (name, _) in enumerate(detectors):
            cr = m["readouts"][name]["task"]["slices"]["all_runs"]["catch_rate"][f"fpr_{fpr}"]
            for k, (arm, label, color) in enumerate(ARMS):
                v, (lo, hi) = cr[arm]["value"], cr[arm]["ci95"]
                x = j + (k - 1) * width
                ax.bar(x, v, width - 0.03, color=color, label=label if j == 0 else None)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="#52514e", capsize=2, lw=1)
                ax.text(x, hi + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#0b0b0b")
        fas = [m["readouts"][n]["task"]["slices"]["all_runs"]["catch_rate"][f"fpr_{fpr}"]["honest_false_alarm_rate"]["value"]
               for n, _ in detectors]
        ax.set_xticks(range(len(detectors)), [f"{d}\n(test false alarms {fa:.3f})" for (_, d), fa in zip(detectors, fas)],
                      fontsize=8)
        ax.set_title(f"Threshold frozen at {float(fpr):.0%} false alarms on validation honest runs", fontsize=10)
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes:
        ax.set_ylabel("Catch rate (share of attack\nruns flagged; higher is better)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Detectors trained on real honest vs real attack (epoch 1), scored on held-out main tasks", fontsize=11)
    fig.tight_layout()
    out = HERE / "figures" / f"catch_rates_{Path(args.metrics).stem}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
