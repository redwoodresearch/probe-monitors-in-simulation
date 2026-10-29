"""Roleplaying AUROC against relative depth (from the latest layer_sweep CSV).

  uv run --with pandas --with matplotlib python experiments/2026-10-06_apollo_qwen_gap/plot_layer_sweep.py

Figures (results/):
- roleplaying_auroc_by_depth_*.png: Apollo's Roleplaying set. Llama-3.3-70B with Apollo's readout (final norm),
  Qwen3-32B with the same readout, and Qwen3-32B's raw block output dashed.
- onpolicy_by_depth_*.png: both readers on Llama-written vs Qwen3-32B-written responses (final-norm readout).
- size_by_depth_*.png: every model, the 8B and 14B ones next to the large ones (final-norm readout).
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"  # categorical slots 1-5, fixed per model
MUTED, GRID = "#52514e", "#e8e7e2"
NAMES = {"llama-3.3-70b": "Llama-3.3-70B", "qwen3-32b": "Qwen3-32B", "llama-3.1-8b": "Llama-3.1-8B", "qwen3-8b": "Qwen3-8B",
         "qwen3-14b": "Qwen3-14B"}
COLOR = {"llama-3.3-70b": BLUE, "qwen3-32b": ORANGE, "llama-3.1-8b": AQUA, "qwen3-8b": YELLOW, "qwen3-14b": MAGENTA}
SETS = {"llama_written_apollo": "Llama-3.3-70B's responses (Apollo's set)", "qwen_written": "Qwen3-32B's own responses"}
FOOT = ("Probe: Apollo's recipe (logistic regression on 306 instructed honest/dishonest statement pairs), refit at every layer. "
        "Shaded: 95% bootstrap interval. Dot: best layer, picked on this test set (optimistic).")


def line(ax, d: pd.DataFrame, color: str, style: str, label: str, note_dy: int):
    if d.empty:  # that model's sweep (or that eval set) is not in this CSV yet
        print(f"  skipped: {label} (no rows)")
        return
    d = d.sort_values("depth")
    ax.fill_between(d["depth"] * 100, d["ci_low"], d["ci_high"], color=color, alpha=0.12, linewidth=0)
    ax.plot(d["depth"] * 100, d["auroc"], style, color=color, linewidth=2, label=label)
    best = d.loc[d["auroc"].idxmax()]
    ax.plot(best["depth"] * 100, best["auroc"], "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"{best['auroc']:.3f} (L{int(best['layer'])})", (best["depth"] * 100, best["auroc"]), textcoords="offset points",
                xytext=(6, note_dy), fontsize=8, color=MUTED)


def frame(ax, sub: pd.DataFrame, title: str):
    ax.axhline(0.5, color="#b5b4ad", linewidth=1)
    null = sub["null95"].median()
    ax.axhline(null, color="#b5b4ad", linewidth=1, linestyle=":")
    ax.text(99, null + 0.005, "label-shuffle 95th percentile", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.axvline(22 / 80 * 100, color="#b5b4ad", linewidth=1, linestyle=":")
    ax.text(22 / 80 * 100 + 1, 0.41, "Apollo's layer\n(22 of 80)", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Relative depth: blocks applied / all blocks (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def finish(fig, axes, title: str, path: Path):
    axes[0].set_ylabel("Roleplaying AUROC (higher is better)")
    axes[-1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    width = int(fig.get_figwidth() * 17)  # characters per line at fontsize 7
    fig.text(0.01, 0.005, textwrap.fill(FOOT, width), fontsize=7, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # The stamp comes from the CSV, so a rerun on the same CSV must not replace a committed figure: create it
    # exclusively ("xb" raises if it exists, atomically). Run analyze_layer_sweep.py for a new CSV first.
    with open(path, "xb") as f:
        fig.savefig(f, format="png", dpi=160)
    plt.close(fig)
    print(f"-> {path}")


def counts(sub: pd.DataFrame) -> str:
    return f"{int(sub['n_deceptive'].iloc[0])} deceptive vs {int(sub['n_honest'].iloc[0])} honest"


def main():
    csv = sorted(RESULTS.glob("layer_sweep_2*.csv"))[-1]
    t = pd.read_csv(csv)
    stamp = csv.stem.removeprefix("layer_sweep_")
    pick = lambda s, m, r: t[(t["label_set"] == s) & (t["model"] == m) & (t["readout"] == r)]

    # 1. The requested figure: Apollo's set; Llama and Qwen with Apollo's readout, Qwen's raw readout dashed.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    s = "llama_written_apollo"
    line(ax, pick(s, "llama-3.3-70b", "apollo"), BLUE, "-", "Llama-3.3-70B, final norm (Apollo's readout)", 6)
    line(ax, pick(s, "qwen3-32b", "apollo"), ORANGE, "-", "Qwen3-32B, final norm", -14)
    line(ax, pick(s, "qwen3-32b", "block_output"), ORANGE, "--", "Qwen3-32B, raw block output", 6)
    frame(ax, t[t["label_set"] == s], f"{SETS[s]}: {counts(t[t['label_set'] == s])}")
    finish(fig, [ax], "Apollo's deception probe at every layer: Llama-3.3-70B vs Qwen3-32B", RESULTS / f"roleplaying_auroc_by_depth_{stamp}.png")

    # 2. On-policy test: each reader on each writer's responses.
    if "qwen_written" in set(t["label_set"]):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
        for ax, s in zip(axes, SETS):
            for m, dy in (("llama-3.3-70b", 6), ("qwen3-32b", -14)):
                line(ax, pick(s, m, "apollo"), COLOR[m], "-", f"{NAMES[m]} reading, final norm", dy)
            frame(ax, t[t["label_set"] == s], f"{SETS[s]}: {counts(t[t['label_set'] == s])}")
        finish(fig, list(axes), "On-policy test: does Qwen3-32B's probe do better on text Qwen wrote?", RESULTS / f"onpolicy_by_depth_{stamp}.png")

    # 3. Size check: every model on Apollo's set (the 8B and 14B models beside the large ones).
    if {"llama-3.1-8b", "qwen3-8b", "qwen3-14b"} & set(t["model"]):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        s = "llama_written_apollo"
        for m, dy in (("llama-3.3-70b", 6), ("qwen3-32b", -14), ("llama-3.1-8b", -14), ("qwen3-8b", 6), ("qwen3-14b", 6)):
            line(ax, pick(s, m, "apollo"), COLOR[m], "-", f"{NAMES[m]}, final norm", dy)
        frame(ax, t[t["label_set"] == s], f"{SETS[s]}: {counts(t[t['label_set'] == s])}")
        finish(fig, [ax], "Model-size check: every model with the same probe recipe", RESULTS / f"size_by_depth_{stamp}.png")


if __name__ == "__main__":
    main()
