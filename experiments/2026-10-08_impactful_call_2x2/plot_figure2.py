"""Figure 2: the off-policy 2x2 as literal 2x2 grids, one per measure of the on-policy plan (amendment 40), from
analyze_pauc_2x2.py's result (local). Rows: trained on real / simulated; columns: tested on real / simulated. Each cell
shows the value and its 95% interval, shaded by value on one sequential scale per grid. Under each grid: simulated
minus real for each training world (the domain gaps) and simulated-trained minus real-trained for each test world (the
transfer contrasts), every gap simulated minus real (amendment 43), with 95% intervals from the same resamples. One
file per reader.

  python3 experiments/2026-10-08_impactful_call_2x2/plot_figure2.py experiments/2026-10-08_impactful_call_2x2/results/pauc_2x2_2026-10-08_171441.json
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
W = ("real", "sim")
NAMES = {"real": "real", "sim": "simulated"}
MEASURES = [("auroc", "AUROC", (0.97, 1.0)), ("pauc_0_5pct", "Standardized pAUC, false alarms 0–5%", (0.85, 1.0)),
            ("catch_1pct_calibrated", "Catch at 1% false alarms (noisy)", (0.6, 1.0))]


def fmt(metric: str, v: float) -> str:
    return f"{v:.1%}" if metric.startswith("catch") else f"{v:.3f}"


def dfmt(metric: str, v: float) -> str:
    return f"{v * 100:+.1f} pts" if metric.startswith("catch") else f"{v:+.4f}"


def contrast_lines(r: dict, metric: str) -> list[str]:
    """Simulated minus real per training world (domain gaps) and per test world (transfer), as the result file signs them."""
    out = []
    for train, key in (("real", "domain gap, trained on real (simulated minus real: real->sim minus real->real)"),
                       ("sim", "domain gap, trained on sim (simulated minus real: sim->sim minus sim->real)")):
        d = r[key][metric]
        out.append(f"trained on {NAMES[train]}, tested simulated − real: {dfmt(metric, d['value'])} [{dfmt(metric, d['ci95'][0])}, {dfmt(metric, d['ci95'][1])}]")
    for key, name in (("transfer, tested on simulated (sim-trained minus real-trained: sim->sim minus real->sim)", "tested on simulated, trained simulated − real"),
                      ("transfer, tested on real (sim-trained minus real-trained: sim->real minus real->real)", "tested on real, trained simulated − real")):
        d = r[key][metric]
        out.append(f"{name}: {dfmt(metric, d['value'])} [{dfmt(metric, d['ci95'][0])}, {dfmt(metric, d['ci95'][1])}]")
    return out


def draw(r: dict, reader: str, source: str, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    for ax, (metric, title, (lo, hi)) in zip(axes, MEASURES):
        grid = [[r[f"{t}->{te}"][metric]["value"] for te in W] for t in W]
        im = ax.imshow(grid, cmap="Blues", vmin=lo, vmax=hi)
        for i, t in enumerate(W):
            for j, te in enumerate(W):
                c = r[f"{t}->{te}"][metric]
                dark = (c["value"] - lo) / (hi - lo) > 0.55
                ax.text(j, i - 0.08, fmt(metric, c["value"]), ha="center", va="center", fontsize=18, weight="bold", color="white" if dark else "black")
                ax.text(j, i + 0.22, f"[{fmt(metric, c['ci95'][0])}, {fmt(metric, c['ci95'][1])}]", ha="center", va="center", fontsize=10,
                        color="white" if dark else "black")
        ax.set_xticks([0, 1], [f"tested on\n{NAMES[w]}" for w in W])
        ax.set_yticks([0, 1], [f"trained on\n{NAMES[w]}" for w in W])
        ax.xaxis.tick_top()
        ax.set_title(title + " (higher is better)", fontsize=11, pad=34)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(title, fontsize=8)
        ax.text(0.5, -0.1, "\n".join(contrast_lines(r, metric)), transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    fig.suptitle(f"Off-policy 2x2: impactful-command probe, {reader} reading Sol's transcripts", fontsize=14, y=0.99)
    fig.text(0.01, 0.01, "Preliminary. Attack runs, the last content token of each command (content call_ends). Catch at 1%: threshold calibrated per "
             "fold on the test world's own read-only calls, in sample (about 5 of about 500 per fold), so noisy.\nCells and contrasts: 95% "
             f"intervals from 1,000 main-task resamples (seed 0), each recalibrating the 1% threshold. Shading: one sequential scale per grid. Data: {source}.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out, dpi=120)
    print(out, out.stat().st_size, "bytes")


def main(path: Path):
    res = json.loads(path.read_text())
    stamp = path.stem.split("pauc_2x2_")[1]
    for reader, tag in (("Qwen3.5-27B", "qwen35"), ("Qwen3-32B", "qwen3-32b")):
        draw(res[reader], reader, path.name, HERE / "results" / f"figure2_2x2_{tag}_{stamp}.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
