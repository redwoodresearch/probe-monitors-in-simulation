"""Minimal figures from a tpr_frozen result (local): per setup, a 2x2 of TPR at the frozen 1%-FPR threshold (with the
FPR it gives in each world), and a log-x ROC of the real-trained probe on real vs simulated calls. Style: the
orchestrator's figs_min template; explanations go in the README, not on the figures.

  uv run --no-project --with matplotlib --with numpy python experiments/2026-10-09_q2_tpr_at_1pct/plot_tpr_figs.py <tpr_frozen json>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
SETUPS = {"off-policy (Qwen3.5-27B reading Sol's runs; refit)": ("offpolicy", "Off-policy: Sol acts, Qwen3.5 reads"),
          "on-policy (Qwen3.5-27B acting; September simulator)": ("onpolicy", "On-policy: Qwen3.5 acts, September simulator")}


def grid(v: dict, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cells = [["real->real", "real->sim"], ["sim->real", "sim->sim"]]
    ax.imshow(np.array([[v["cells"][c]["TPR"] for c in row] for row in cells]), cmap="Blues", vmin=0.4, vmax=1.0)
    for i, row in enumerate(cells):
        for j, c in enumerate(row):
            x = v["cells"][c]
            tci = next(val for k, val in x.items() if k.startswith("TPR 95"))
            fci = next(val for k, val in x.items() if k.startswith("FPR 95"))
            ax.text(j, i - 0.2, f"{100 * x['TPR']:.1f}%", ha="center", va="center", fontsize=20, fontweight="bold", color="white")
            ax.text(j, i + 0.06, f"[{100 * tci[0]:.1f}, {100 * tci[1]:.1f}]", ha="center", va="center", fontsize=10, color="white")
            ax.text(j, i + 0.3, f"FPR {100 * x['FPR']:.1f}% [{100 * fci[0]:.1f}, {100 * fci[1]:.1f}]", ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks([0, 1], ["tested on\nreal", "tested on\nsimulated"], fontsize=11)
    ax.set_yticks([0, 1], ["trained on\nreal", "trained on\nsimulated"], fontsize=11)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{title}\nTPR at 1% FPR, threshold frozen on the training world", fontsize=11.5, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc(v: dict, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    lo = 1 / min(v["cells"]["real->real"]["negatives"], v["cells"]["real->sim"]["negatives"])
    for w, col, name in (("real", "#2f6db5", "tested on real"), ("sim", "#c4661a", "tested on simulated")):
        pts = np.array(v["ROC, real-trained probe (FPR, TPR)"][w])
        pts = pts[pts[:, 0] >= lo]
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2, label=name)
        c = v["cells"][f"real->{w}"]
        ax.plot(c["FPR"], c["TPR"], "o", color=col, ms=8)
        ax.annotate(f"TPR {100 * c['TPR']:.1f}%, FPR {100 * c['FPR']:.2f}%", (c["FPR"], c["TPR"]), xytext=(8, -14 if w == "sim" else 8),
                    textcoords="offset points", fontsize=9, color=col)
    ax.axvline(0.01, color="0.6", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(lo, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate (log scale; lower is better)", fontsize=10.5)
    ax.set_ylabel("True positive rate (higher is better)", fontsize=10.5)
    ax.set_title(f"{title}\nreal-trained probe; dots: the frozen threshold", fontsize=11.5)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(res_path: Path):
    res, stamp = json.loads(res_path.read_text()), res_path.stem.removeprefix("tpr_frozen_")
    for key, (tag, title) in SETUPS.items():
        for kind, fn in (("tpr_2x2", grid), ("roc_logx", roc)):
            out = HERE / "results" / f"{kind}_{tag}_{stamp}.png"
            fn(res[key], title, out)
            print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
