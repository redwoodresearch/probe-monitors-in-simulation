"""Does a probe trained by reading Sol's runs work on Qwen's own calls, by layer? One PNG per metric. $0, local.

  python3 plot_transfer_layers.py results/offpolicy_checks_<stamp>.json results/offpolicy_transfer_fixed_layers_<stamp>.json \
      results/onpolicy_2x2_analysis_rerun_strict_september_<stamp>.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
KEY = "(b) on-policy real (attack), all scored calls"


def main(checks, fixed, onpol):
    C, F, A = (json.loads(Path(p).read_text()) for p in (checks, fixed, onpol))
    bars = [(f"Trained reading Sol,\nlayer {C['transfer']['probe']['layer']}", C["transfer"][KEY], "#9aa3ad")]
    for k, v in sorted(F["transfer_by_layer"].items(), key=lambda kv: kv[1]["probe"]["layer"]):
        bars.append((f"Trained reading Sol,\nlayer {v['probe']['layer']}", v[KEY], "#9aa3ad"))
    rr = A["cells"]["real->real"]
    bars.append(("Trained on Qwen acting\n(20 tasks, figure 3)", {m: rr[m]["value"] for m in ("auroc", "pauc")} | {f"{m}_ci95": rr[m]["ci95"] for m in ("auroc", "pauc")}, "#2f6db5"))
    for m, label in (("auroc", "AUROC (1 = perfect)"), ("pauc", "Ranking at 0–5% false alarms\n(standardized pAUC, 1 = perfect)")):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for i, (name, r, c) in enumerate(bars):
            v, (lo, hi) = r[m], r[f"{m}_ci95"]
            ax.bar(i, v, color=c, width=0.6)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="black", capsize=6, lw=1.5)
            ax.text(i, hi + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.set_ylim(0.6 if m == "auroc" else 0.5, 1.06)
        ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=10)
        ax.set_ylabel(label + "\n(higher is better)", fontsize=11)
        ax.set_title("Scored on Qwen's own calls: does a probe trained by reading carry over?", fontsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        out = HERE / "figures" / f"transfer_layers_{m}_{Path(fixed).stem.split('_')[-2]}_{Path(fixed).stem.split('_')[-1]}.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=160)
        print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
