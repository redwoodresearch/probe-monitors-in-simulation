"""Per-layer AUROC curves for the matched pairs: frozen Q1-recipe probe and fresh within-pairs probe, at both readouts,
with 95% intervals (resampling main tasks) and each probe's label-shuffle null p95. Reads results/metrics_<stamp>.json.

  uv run --no-project --with matplotlib python experiments/2026-10-06_matched_step_pairs/plot_layers.py results/metrics_<stamp>.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {"frozen": "#2a78d6", "fresh": "#eb6834"}  # reference palette slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main(path: Path):
    M = json.loads(path.read_text())["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, readout, title in zip(axes, ("output_end", "call_end"),
                                  ("output_end of step k (the output itself)", "call_end of the next call (same real command)")):
        for name, curve, null_key, chosen in (
                ("frozen", M["probe_frozen_q1"][readout]["curve"], "null_training_label_shuffle",
                 M["probe_frozen_q1"][readout]["chosen_layer"]),
                ("fresh", M["probe_fresh"][readout]["curve"], "null_within_pair_swap", None)):
            layers = sorted(curve, key=int)
            x = [int(l) for l in layers]
            a = [curve[l]["all"]["auroc"] for l in layers]
            lo = [curve[l]["all"]["auroc_ci"][0] for l in layers]
            hi = [curve[l]["all"]["auroc_ci"][1] for l in layers]
            p95 = [curve[l][null_key]["p95"] for l in layers]
            label = "frozen Q1-recipe probe (trained on cohorts)" if name == "frozen" else "fresh probe (within pairs, leave-env-out)"
            ax.fill_between(x, lo, hi, color=SERIES[name], alpha=0.15, linewidth=0)
            ax.plot(x, a, color=SERIES[name], linewidth=2, label=label)
            ax.plot(x, p95, color=SERIES[name], linewidth=1, linestyle=":", label=f"{name}: label-shuffle null p95")
            if chosen is not None:
                ax.plot([chosen], [curve[str(chosen)]["all"]["auroc"]], "o", color=SERIES[name], markersize=8)
                ax.annotate(f"L{chosen} (cohort CV)\n{curve[str(chosen)]['all']['auroc']:.3f}", (chosen, curve[str(chosen)]["all"]["auroc"]),
                            textcoords="offset points", xytext=(6, 10), color=INK, fontsize=8)
        ax.axhline(0.5, color=MUTED, linewidth=1)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("Qwen3-32B layer", color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUROC, real vs simulated output\n(higher = easier to tell apart; 0.5 = chance)", color=MUTED)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Matched-step pairs (200): same real history, real vs simulated output at step k; shaded = 95% CI",
                 fontsize=11, color=INK)
    fig.tight_layout()
    out = path.parent / f"figure_layers_{path.stem.removeprefix('metrics_')}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
