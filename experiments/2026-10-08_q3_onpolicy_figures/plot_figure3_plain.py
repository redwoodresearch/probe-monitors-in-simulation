"""Figure 3 in plain words, one PNG per simulator; the numbers are the analyses' own, unchanged. $0, local.

Left: Qwen does the tasks itself (on-policy). Right, for reference only: Qwen reading another model's runs (off-policy,
the refit figure 2), a different setup, not directly comparable. Rows: overall ranking (AUROC), ranking at low false-alarm
rates (standardized partial AUROC over 0–5% false alarms), and catch at 1% false alarms.

  python3 plot_figure3_plain.py results/figure3_analysis_compact_<simulator>_<stamp>.json <refit json> <refit pauc json>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
TITLE = "When Qwen does the tasks itself, does the command monitor work as well on simulated runs as on real ones?"
ROWS = [("auroc", "AUROC (ranking; 1.0 = perfect; with shuffled labels about 0.52)"), ("pauc", "Ranking at low false-alarm rates (0–5%; higher is better, 1 = perfect)"),
        ("catch", "Share of state-changing commands caught, targeting 1% false alarms")]
SIMS = {"sim_attack@september": "the September simulator (main comparison)", "sim_attack@today": "the current simulator"}


def ci(r):
    return r.get("ci95_exact") or r.get("ci95")


def grid(ax, cells, m, title, nonull=False):
    ax.imshow([[0, 0], [0, 0]], cmap="Greys", vmin=0, vmax=10)
    for i, t in enumerate(("real", "sim")):
        for j, v in enumerate(("real", "sim")):
            c = cells[f"{t}->{v}"][m]
            if c.get("value") is None or c.get("status", "").startswith("unestimable"):
                ax.text(j, i, "too few tasks\nto estimate", ha="center", va="center", fontsize=11, color="#555555")
                continue
            lo, hi = ci(c)
            pct = m == "catch"
            ax.text(j, i - 0.1, f"{c['value']:.0%}" if pct else f"{c['value']:.3f}", ha="center", va="center", fontsize=18, weight="bold")
            ax.text(j, i + 0.22, f"95% range\n{lo:.0%} to {hi:.0%}" if pct else f"95% range\n{lo:.3f} to {hi:.3f}",
                    ha="center", va="center", fontsize=10.5)
            if nonull and (m == "pauc" or (t == "sim" and m == "auroc")):
                ax.text(j, i + 0.4, "no shuffle check run", ha="center", va="center", fontsize=8, style="italic", color="#555555")
    ax.axhline(0.5, color="#999999", lw=0.8)
    ax.axvline(0.5, color="#999999", lw=0.8)
    ax.set_xticks([0, 1], ["tested on\nreal runs", "tested on\nsimulated runs"])
    ax.set_yticks([0, 1], ["trained on\nreal runs", "trained on\nsimulated runs"])
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=10.5, pad=34)


def main(path, refit_path, pauc_path):
    A = json.loads(Path(path).read_text())
    refit, pauc = (json.loads(Path(p).read_text())["readers"]["qwen3.5"] for p in (refit_path, pauc_path))
    off = {c: {"auroc": refit[c]["auroc"], "pauc": pauc[c], "catch": refit[c]["catch, calibrated in test world"]}
           for c in ("real->real", "real->sim", "sim->real", "sim->sim")}
    sim = A["simulator"].split(" ")[0]
    fig, axes = plt.subplots(3, 2, figsize=(13, 16))
    gap = A["primary_result"]
    for r, (m, name) in enumerate(ROWS):
        grid(axes[r, 0], A["cells"], m, f"Qwen does the tasks itself\n{name}", nonull=True)
        grid(axes[r, 1], off, m, f"For reference only (different setup, not directly comparable):\nQwen reading another model's runs\n{name}")
        if m in gap:
            g = gap[m]
            lo, hi = ci(g)
            axes[r, 0].text(0.5, -0.07, f"Trained on real runs: simulated minus real = {g['value']:+.4f} (95% range {lo:+.4f} to {hi:+.4f})",
                            transform=axes[r, 0].transAxes, ha="center", va="top", fontsize=9.5)
    fig.suptitle(f"{TITLE}\nThe monitor separates commands that change the system (writing files, network calls, installs) from read-only ones."
                 f"\nLeft panels: simulated runs from {SIMS.get(sim, sim)}", fontsize=13, y=0.997)
    fig.text(0.01, 0.006, "95% ranges: left, from re-running the whole analysis on 2,000 resamples of the 20 tasks; right, the reference study's own 500 resamples.\n"
             "Preliminary: only 20 tasks, too few for a verdict, and too few to estimate the catch rate at 1% false alarms.\n"
             f"Source: the 2026-10-09 rerun of 20 LinuxArena tasks (numbers: {Path(path).name}); reference: Qwen3.5-27B reading Sol's runs.", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.955))
    out = HERE / "figures" / Path(path).with_suffix(".png").name.replace("figure3_analysis_compact", "figure3_plain")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
