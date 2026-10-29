"""The two off-policy checks as bars, one PNG per metric (AUROC, standardized pAUC). $0, local. Explanations go in the
text beside the figure, not in it.

  python3 plot_offpolicy_checks.py results/offpolicy_checks_<stamp>.json results/onpolicy_2x2_analysis_rerun_strict_september_<stamp>.json \
      <refit json> <refit pauc json>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
NAMES = ["Reads Sol's runs,\ntrained on 522 tasks", "Reads Sol's runs,\ntrained on 20 tasks", "Trained reading Sol,\ntested on Qwen acting",
         "Qwen acting,\ntrained on 20 tasks"]
COLS = ["#9aa3ad", "#9aa3ad", "#2f6db5", "#2f6db5"]


def main(checks, onpol, refit, pauc):
    C, A = json.loads(Path(checks).read_text()), json.loads(Path(onpol).read_text())
    R, P = (json.loads(Path(p).read_text())["readers"]["qwen3.5"] for p in (refit, pauc))
    s, b = C["size"]["summary"], C["transfer"]["(b) on-policy real (attack), all scored calls"]
    rr = A["cells"]["real->real"]
    for m, label in (("auroc", "AUROC (1 = perfect)"), ("pauc", "Ranking at 0–5% false alarms\n(standardized pAUC, 1 = perfect)")):
        ref = R["real->real"]["auroc"] if m == "auroc" else P["real->real"]
        bars = [(ref["value"], *ref["ci95"]), (s[f"{m}_median"], *s[f"{m}_5_95"]), (b[m], *b[f"{m}_ci95"]), (rr[m]["value"], *rr[m]["ci95"])]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for i, ((v, lo, hi), c) in enumerate(zip(bars, COLS)):
            ax.bar(i, v, color=c, width=0.6)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="black", capsize=6, lw=1.5)
            ax.text(i, hi + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
        lo_y = 0.6 if m == "auroc" else 0.5
        ax.set_ylim(lo_y, 1.06)
        ax.set_xticks(range(4), NAMES, fontsize=10)
        ax.set_ylabel(label + "\n(higher is better)", fontsize=11)
        ax.set_title("Why is the command monitor weaker when Qwen acts?", fontsize=13)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        out = HERE / "figures" / f"offpolicy_checks_{m}_{'_'.join(Path(checks).stem.split('_')[-2:])}.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=160)
        print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
