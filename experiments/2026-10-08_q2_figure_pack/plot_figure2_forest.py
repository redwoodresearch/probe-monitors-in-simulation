"""Figure 2b: the refit's simulated-minus-real gaps as a forest plot, one file per reader (local). For each measure
(AUROC, standardized pAUC 0-5%, catch at 1% calibrated on a held-out fold of the test world), the four contrasts as
points with 95% intervals and a zero line. Same inputs and signs as plot_figure2_refit.py, whose helpers it uses.

  python3 experiments/2026-10-08_q2_figure_pack/plot_figure2_forest.py <refit json> <refit pauc json>
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from plot_figure2_refit import CONTRASTS, HERE, MEASURES, contrast

UNITS = {"auroc": ("difference in ranking", 1), "pauc": ("difference in low-false-alarm ranking", 1), "catch": ("difference in commands flagged, points", 100)}


GROUPS = [("Changing the test runs from real to simulated (same training)", CONTRASTS[:2], "right: better on simulated test runs"),
          ("Changing the training runs from real to simulated (same test runs)", CONTRASTS[2:], "right: the simulated-trained monitor is better")]


def draw(refit: dict, pauc: dict, reader: str, sources: str, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.0), sharey="row")
    for row, (heading, contrasts, direction) in enumerate(GROUPS):
        for ax, (metric, title, _) in zip(axes[row], MEASURES):
            label, scale = UNITS[metric]
            for y, (name, key, sign, pkey) in enumerate(contrasts):
                v, lo, hi = (scale * x for x in contrast(refit, pauc, metric, key, sign, pkey))
                ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color="tab:blue", capsize=5)
                d = 1 if scale > 1 else 4
                ax.text(0.98, y - 0.3, f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]", ha="right", va="center", fontsize=8.5,
                        transform=mtransforms.blended_transform_factory(ax.transAxes, ax.transData))
            ax.axvline(0, color="black", lw=1)
            ax.set_title(f"{title}\n{heading}" if row == 0 else heading, fontsize=9.5)
            ax.set_xlabel(f"{label}\n({direction})", fontsize=8.5)
            ax.margins(x=0.15)
        axes[row][0].set_yticks(range(len(contrasts)), [c[0].split(":")[0] for c in contrasts], fontsize=9)
        axes[row][0].set_ylim(len(contrasts) - 0.4, -0.8)
    a, p = (contrast(refit, pauc, m, *CONTRASTS[0][1:]) for m in ("auroc", "pauc"))
    fig.suptitle(f"How much does the command monitor's performance differ between simulated and real runs? ({reader}, simulated minus real)\n"
                 f"Trained on real runs, simulated minus real test runs: ranking {a[0]:+.4f} [{a[1]:+.4f}, {a[2]:+.4f}]; "
                 f"ranking at low false-alarm rates {p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}]", fontsize=12)
    sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    fig.text(0.01, 0.01, "Preliminary. Points: simulated minus real; bars: 95% intervals from 500 resamples of tasks, retraining the monitor each time. "
             "An interval that includes zero bounds the gap; it does not show the gap is zero.\nThe monitor reads runs that another model (Sol) wrote, "
             f"not runs it wrote itself.\nData: {sources}; script at commit {sha}.", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    if out.exists():
        raise FileExistsError(out)
    fig.savefig(out, dpi=120)
    print(out, out.stat().st_size, "bytes")


def main(refit_path: Path, pauc_path: Path):
    refit, pauc = json.loads(refit_path.read_text()), json.loads(pauc_path.read_text())
    for reader, name, tag in (("qwen3.5", "Qwen3.5-27B", "qwen35"), ("qwen3-32b", "Qwen3-32B", "qwen3-32b")):
        if refit["readers"][reader]["fingerprint"] != pauc["readers"][reader]["fingerprint"]:
            raise ValueError(f"{reader}: the pAUC file was computed from other replicates")
        draw(refit["readers"][reader], pauc["readers"][reader], name, f"{refit_path.name}, {pauc_path.name}",
             HERE / "results" / f"figure2b_forest_{tag}_{refit_path.stem.removeprefix('refit_')}_plain.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
