"""The attack detector's primary cell (layer 51, tool-output readout) as four bars: does a detector trained by reading
Sol's runs work on Qwen's own runs, real and simulated? $0, local; numbers from analyze_attack_detector.py's output.

  python3 plot_attack_detector.py results/attack_detector_analysis_<stamp>.json [results/attack_detector_cohort_<stamp>.json]

With the cohort file, the reading bars use the same 20 pilot main tasks as the acting bars. Each bar is annotated with
its label-shuffle null's p-value, or "no null run".
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent


def main(path, cohort=None):
    c = json.loads(Path(path).read_text())["cells"]["L51_output_end"]
    o, w, null = c["off_policy_reference"], c["worlds"], c.get("null", {})
    if cohort:
        k = json.loads(Path(cohort).read_text())["pairs"]
        off = [("Reads Sol's runs,\nreal\n(same 20 tasks)", k["real e1 vs real honest"]["auroc"], k["real e1 vs real honest"]["ci95"]),
               ("Reads Sol's runs,\nsimulated\n(same 20 tasks)", k["sim attack vs sim honest"]["auroc"], k["sim attack vs sim honest"]["ci95"])]
    else:
        off = [("Reads Sol's runs,\nreal", o["real e1 vs real honest"], o["real e1 vs real honest_ci95"]),
               ("Reads Sol's runs,\nsimulated", o["sim attack vs sim honest"], o["sim attack vs sim honest_ci95"])]
    bars = [(n, v, ci, "#9aa3ad", "no null run") for n, v, ci in off] + [
        ("Qwen acting,\nreal", w["real"]["auroc"], w["real"]["auroc_ci95"], "#2f6db5", f"null p = {null['real']['p']:.3f}" if "real" in null else "no null run"),
        ("Qwen acting,\nsimulated (September)", w["september"]["auroc"], w["september"]["auroc_ci95"], "#2f6db5",
         f"null p = {null['september']['p']:.3f}" if "september" in null else "no null run")]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (name, v, (lo, hi), col, note) in enumerate(bars):
        ax.bar(i, v, color=col, width=0.6)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="black", capsize=6, lw=1.5)
        ax.text(i, hi + 0.035, f"{v:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.text(i, hi + 0.012, note, ha="center", va="bottom", fontsize=8.5, color="#444444")
    ax.axhline(0.5, color="#888888", lw=1, ls="--")
    ax.text(3.45, 0.505, "0.5", ha="right", va="bottom", fontsize=9, color="#666666")
    ax.set_ylim(0.4, 1.1)
    ax.set_xticks(range(4), [b[0] for b in bars], fontsize=10)
    ax.set_ylabel("Attack vs honest runs, AUROC\n(1 = perfect; higher is better)", fontsize=11)
    ax.set_title("Does an attack detector trained by reading work when Qwen acts?", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = HERE / "figures" / f"attack_detector_{'_'.join(Path(path).stem.split('_')[-2:])}{'_cohort' if cohort else ''}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
