"""Figure 3 for the meeting: two PNGs from the committed analyses. $0, local.

(a) The 2×2 grids (AUROC and standardized pAUC): on-policy with September's simulator (primary) beside off-policy (final
figure 2, refit), value and 95% interval per cell.
(b) A forest plot of the gaps, simulated − real (the probe trained on real runs; tested on simulated minus tested on real),
with 95% intervals: on-policy September, on-policy today, off-policy (refit, Qwen3.5-27B reader), one panel per measure
on a shared axis, with the borrowed ±0.03 band drawn for reference (no verdict at pilot size).
Run names, files and amendments go in the footer.

  python3 plot_figure3_meeting.py <analysis september> <analysis today> <refit json> <refit pauc json>
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
TITLE = "Does the probe still work the same when Qwen does the tasks itself?"
CELLS = ("real->real", "real->sim", "sim->real", "sim->sim")


def ci(r):
    return r.get("ci95_exact") or r["ci95"]  # unrounded bounds when the analysis has them


def load(sep_path, today_path, refit_path, pauc_path):
    sep, today = (json.loads(Path(p).read_text()) for p in (sep_path, today_path))
    refit, pauc = (json.loads(Path(p).read_text())["readers"]["qwen3.5"] for p in (refit_path, pauc_path))
    if refit["fingerprint"] != pauc["fingerprint"]:
        raise ValueError("the refit and pAUC files come from different replicates")
    off_cells = {c: {"auroc": refit[c]["auroc"], "pauc": pauc[c]} for c in CELLS}
    g = refit["domain gap, trained on real (real->real minus real->sim)"]["auroc"]  # stored real − simulated: negate
    off_gaps = {"auroc": {"value": -g["value"], "ci95": sorted(-x for x in g["ci95"])},
                "pauc": pauc["domain gap, trained on real: real->sim − real->real"]}  # already simulated − real
    return sep, today, off_cells, off_gaps


def wrap(text, width=190):
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def footer(sep, today, files):
    return wrap(f"On-policy: Qwen3.5-27B acting in real and simulated LinuxArena, attack runs, {sep['summary']['analyzable_pairs']} tasks (rerun "
            f"2026-10-09_000523_rerun_strict); readout: the last token of each command. Off-policy: Qwen3.5-27B reading Sol's transcripts "
            f"(final figure 2, refit).\nOn-policy intervals: {sep['draws']} task resamples, each refitting the probe, re-picking layer and C, "
            f"recalibrating. Catch at 1% false alarms: unestimable at this size (a calibration fold holds under 100 read-only calls).\n"
            f"No verdict at pilot size. Files: {', '.join(files)}.\nAmendments: " + "; ".join(f"{k}: {v}" for k, v in sep["amendments"].items()))


def grids(sep, off_cells, out, foot):
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for row, (m, name, lo, hi) in enumerate((("auroc", "AUROC", 0.9, 1.0), ("pauc", "Partial AUROC, false alarms 0–5%, standardized", 0.7, 1.0))):
        for col, (cells, who) in enumerate(((sep["cells"], "Qwen acts (on-policy), September simulator"),
                                            (off_cells, "Reference: Qwen reads Sol's runs (off-policy)\nnot a matched comparison: other pipeline, data and tasks"))):
            ax = axes[row, col]
            ax.imshow([[0, 0], [0, 0]], cmap="Greys", vmin=0, vmax=10)  # neutral cells: no colour scale to misread
            for i, t in enumerate(("real", "sim")):
                for j, v in enumerate(("real", "sim")):
                    c = cells[f"{t}->{v}"][m]
                    lo_, hi_ = ci(c)
                    ax.text(j, i - 0.08, f"{c['value']:.3f}", ha="center", va="center", fontsize=18, weight="bold")
                    ax.text(j, i + 0.25, f"95% interval\n[{lo_:.3f}, {hi_:.3f}]", ha="center", va="center", fontsize=9.5)
            for k in (0.5,):
                ax.axhline(k, color="#999999", lw=0.8)
                ax.axvline(k, color="#999999", lw=0.8)
            ax.set_xticks([0, 1], ["tested on\nreal runs", "tested on\nsimulated runs"])
            ax.set_yticks([0, 1], ["trained on\nreal runs", "trained on\nsimulated runs"])
            ax.xaxis.tick_top()
            ax.set_title(f"{who}\n{name} (higher is better)", fontsize=11, pad=34)
    fig.suptitle(TITLE + "\n20-task pilot: no verdict. Catch at 1% false alarms: not estimable at this size.", fontsize=14, y=0.995)
    fig.text(0.01, 0.005, foot, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(out, dpi=120)
    return out


def forest(sep, today, off_gaps, out, foot):
    rows = [("Qwen acts, September simulator (primary)", sep["primary_result"]),
            ("Qwen acts, current simulator (includes sims 4b31bd30)", today["primary_result"]),
            ("Reference, not matched: Qwen reads Sol's runs (off-policy)", off_gaps)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharey=True)
    for ax, (m, name) in zip(axes, (("auroc", "AUROC"), ("pauc", "Partial AUROC, false alarms 0–5%, standardized"))):
        ax.axvspan(-0.03, 0.03, color="#eeeeee", zorder=0, label="reference ±0.03 only; no equivalence verdict")
        ax.axvline(0, color="black", lw=0.8)
        for i, (label, r) in enumerate(rows):
            v, (lo, hi) = r[m]["value"], ci(r[m])
            y = len(rows) - 1 - i
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt=("o", "o", "s")[i], color=("#1f4e79", "#4c8cbf", "#444444")[i], capsize=5, ms=7)
            ax.text(v, y - 0.22, f"{v:+.4f}  [{lo:+.4f}, {hi:+.4f}]", fontsize=9.5, ha="center", va="top",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
        ax.set_ylim(-0.7, len(rows) - 0.5)
        ax.set_xlim(-0.13, 0.19)
        ax.set_xlabel("simulated − real  (0 = equal metric values; negative = lower on simulated)", fontsize=9.5)
        ax.set_title(name, fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.905), fontsize=8.5, frameon=False)
    fig.suptitle(TITLE + "\nProbe trained on real runs: simulated − real, 95% intervals. 20-task pilot: no verdict.", fontsize=13)
    fig.text(0.01, 0.005, foot, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.86))
    fig.savefig(out, dpi=130)
    return out


def main(sep_path, today_path, refit_path, pauc_path):
    sep, today, off_cells, off_gaps = load(sep_path, today_path, refit_path, pauc_path)
    files = [Path(p).name for p in (sep_path, today_path, refit_path, pauc_path)]
    foot = footer(sep, today, files)
    stamp = Path(sep_path).stem.rsplit("_", 2)[-2] + "_" + Path(sep_path).stem.rsplit("_", 1)[-1]
    (HERE / "figures").mkdir(exist_ok=True)
    for p in (grids(sep, off_cells, HERE / "figures" / f"figure3_meeting_grids_{stamp}.png", foot),
              forest(sep, today, off_gaps, HERE / "figures" / f"figure3_meeting_forest_{stamp}.png", foot)):
        print("->", p)


if __name__ == "__main__":
    main(*sys.argv[1:5])
