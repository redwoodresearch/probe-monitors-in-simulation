"""Figure 2 (final): the off-policy 2x2 for the impactful-command probe from the refit bootstrap, as literal 2x2 grids
for AUROC, standardized pAUC 0-5% and catch at 1% false alarms, one file per reader (local).

Inputs, from experiments/2026-10-08_impactful_refit_bootstrap (results PR stacked on #197; both on S3 per its
results/manifest_v1.json): refit_<stamp>.json (AUROC, catch; it stores the domain gaps and the real-target transfer as
real minus simulated, negated here) and refit_pauc_<stamp>.json (pAUC, already simulated minus real). Every contrast is
drawn simulated minus real (amendment 43).

  python3 experiments/2026-10-08_q2_figure_pack/plot_figure2_refit.py <refit json> <refit pauc json> [<refit counts json>]
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
W = ("real", "sim")
NAMES = {"real": "real runs", "sim": "simulated runs"}
MEASURES = [("auroc", "AUROC (ranking; 1.0 = perfect, 0.5 = chance)", (0.97, 1.0)),
            ("pauc", "Ranking at low false-alarm rates (0–5%)", (0.85, 1.0)),
            ("catch", "Share of state-changing commands flagged\nwhen 1% of read-only commands are flagged", (0.6, 1.0))]
REFIT_KEYS = {"auroc": "auroc", "catch": "catch, calibrated in test world"}
# (label under the grid, refit key, sign that makes it simulated minus real, pAUC key)
CONTRASTS = [("trained on real runs: simulated minus real test runs", "domain gap, trained on real (real->real minus real->sim)", -1,
              "domain gap, trained on real: real->sim − real->real"),
             ("trained on simulated runs: simulated minus real test runs", "domain gap, trained on sim (sim->real minus sim->sim)", -1,
              "domain gap, trained on sim: sim->sim − sim->real"),
             ("tested on simulated runs: simulated-trained minus real-trained", "transfer loss on the simulated target (sim->sim minus real->sim)", 1,
              "transfer, tested on sim: sim->sim − real->sim"),
             ("tested on real runs: simulated-trained minus real-trained", "transfer loss on the real target (real->real minus sim->real)", -1,
              "transfer, tested on real: sim->real − real->real")]


def fmt(metric: str, v: float) -> str:
    return f"{v:.1%}" if metric == "catch" else f"{v:.3f}"


def cell(refit: dict, pauc: dict, name: str, metric: str) -> dict:
    return pauc[name] if metric == "pauc" else refit[name][REFIT_KEYS[metric]]


def contrast(refit: dict, pauc: dict, metric: str, key: str, sign: int, pkey: str) -> tuple:
    if metric == "pauc":
        return pauc[pkey]["value"], *pauc[pkey]["ci95"]
    c = refit[key][REFIT_KEYS[metric]]
    return sign * c["value"], *sorted(sign * x for x in c["ci95"])


def answer(refit: dict, pauc: dict) -> str:
    """The headline as numbers: the monitor trained on real runs, tested on simulated minus real runs."""
    a, p = (contrast(refit, pauc, m, *CONTRASTS[0][1:]) for m in ("auroc", "pauc"))
    return (f"Answer: the differences are small, with uncertainty.\nMonitor trained on real runs, simulated minus real test runs: "
            f"ranking (AUROC) {a[0]:+.4f} [{a[1]:+.4f}, {a[2]:+.4f}]; ranking at low false-alarm rates {p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}].")


def data_line(reader: str, counts: dict) -> str:
    r, s = counts["real-attack-e1"], counts["sim-attack"]
    return (f"The monitor reads {reader}'s internal activations on attack runs that another model (Sol) wrote. Real: {r['main tasks']} tasks, "
            f"{r['runs']} runs, {r['impactful calls']:,} state-changing and {r['read-only calls']:,} read-only commands. Simulated: {s['main tasks']} tasks, "
            f"{s['runs']} runs, {s['impactful calls']:,} and {s['read-only calls']:,}.\nState-changing: any command that is not clearly read-only.")


def draw(refit: dict, pauc: dict, reader: str, sources: str, out: Path, counts: dict | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
    for ax, (metric, title, (lo, hi)) in zip(axes, MEASURES):
        grid = [[cell(refit, pauc, f"{t}->{te}", metric)["value"] for te in W] for t in W]
        im = ax.imshow(grid, cmap="Blues", vmin=lo, vmax=hi)
        for i, t in enumerate(W):
            for j, te in enumerate(W):
                c = cell(refit, pauc, f"{t}->{te}", metric)
                color = "white" if (c["value"] - lo) / (hi - lo) > 0.55 else "black"
                ax.text(j, i - 0.08, fmt(metric, c["value"]), ha="center", va="center", fontsize=18, weight="bold", color=color)
                ax.text(j, i + 0.22, f"[{fmt(metric, c['ci95'][0])}, {fmt(metric, c['ci95'][1])}]", ha="center", va="center", fontsize=10, color=color)
        ax.set_xticks([0, 1], [f"tested on\n{NAMES[w]}" for w in W])
        ax.set_yticks([0, 1], [f"trained on\n{NAMES[w]}" for w in W])
        ax.xaxis.tick_top()
        ax.set_title(f"{title}\n(higher is better)", fontsize=11, pad=34)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # the differences themselves are in the forest plot (figure 2b)
    fig.suptitle(f"Does the command monitor work as well on simulated runs as on real ones? ({reader})", fontsize=13, y=0.99)
    fig.text(0.5, 0.955, answer(refit, pauc), ha="center", va="top", fontsize=10.5, weight="bold")
    if counts:
        fig.text(0.5, 0.895, data_line(reader, counts), ha="center", va="top", fontsize=8.5)
    sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    fig.text(0.01, 0.01, "Preliminary. Each cell: the value and its 95% interval from 500 resamples of tasks, retraining the monitor each time.\n"
             "The monitor reads runs that another model (Sol) wrote, not runs it wrote itself.\n"
             f"Data: {sources}; script at commit {sha}.", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.84))
    if out.exists():
        raise FileExistsError(out)
    fig.savefig(out, dpi=120)
    print(out, out.stat().st_size, "bytes")


def main(refit_path: Path, pauc_path: Path, counts_path: Path | None = None):
    refit, pauc = json.loads(refit_path.read_text()), json.loads(pauc_path.read_text())
    counts = json.loads(counts_path.read_text()) if counts_path else {}
    stamp = refit_path.stem.removeprefix("refit_")
    (HERE / "results").mkdir(exist_ok=True)
    for reader, name, tag in (("qwen3.5", "Qwen3.5-27B", "qwen35"), ("qwen3-32b", "Qwen3-32B", "qwen3-32b")):
        if refit["readers"][reader]["fingerprint"] != pauc["readers"][reader]["fingerprint"]:
            raise ValueError(f"{reader}: the pAUC file was computed from other replicates")
        if counts and {k: v for k, v in counts[reader].items() if k != "rows_sha256"} != {
                a: {"impactful calls": c["impactful"], "read-only calls": c["read-only"]} | {k: counts[reader][a][k] for k in ("main tasks", "runs")}
                for a, c in refit["readers"][reader]["analyzed calls"].items()}:
            raise ValueError(f"{reader}: the counts file's calls differ from the refit's analyzed calls")
        draw(refit["readers"][reader], pauc["readers"][reader], name, f"{refit_path.name}, {pauc_path.name}",
             HERE / "results" / f"figure2_refit_{tag}_{stamp}_plain.png", counts.get(reader))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
