"""The "what a drop in standardized pAUC means" table as a figure, from analyze_pauc_meaning.py's result (#248, branch
aswin/q2-pauc-meaning, experiments/2026-10-08_q2_pauc_meaning/results/pauc_meaning_2026-10-08_175417.json) (local).

  python3 experiments/2026-10-08_q2_figure_pack/plot_pauc_meaning.py <pauc_meaning json>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def main(path: Path):
    d = json.loads(path.read_text())
    b = d["before"]
    rows = [["none (as measured)", "", f"{b['pauc']:.3f}", f"{b['call catch']:.1%}", f"{b['run catch']:.1%}"]]
    for a in d["after"]:
        rows.append([f"−{a['drop']:.2f}, {a['how']}", a["size"], f"{a['pauc']:.3f}",
                     f"{a['call catch']:.1%} ({(a['call catch'] - b['call catch']) * 100:+.1f} pts)",
                     f"{a['run catch']:.1%} ({(a['run catch'] - b['run catch']) * 100:+.1f} pts)"])
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["pAUC drop, how", "size of the change", "pAUC after", "impactful calls caught", "attack runs caught"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9.5)
    t.scale(1, 1.55)
    for (r, _), c in t.get_celld().items():
        if r == 0:
            c.set_text_props(weight="bold")
        elif rows[r - 1][0].startswith("−0.03"):
            c.set_facecolor("#e8f0fb")
    ax.set_title("Illustrative score perturbations: what would a 0.03 drop in partial AUROC cost in impactful calls caught?\n"
                 "(hypothetical scenarios on one fixed set of scores, not measured losses)", fontsize=11.5)
    fig.text(0.01, 0.01, f"PRELIMINARY (not yet reviewed by Aswin). Off-policy, Qwen3.5-27B, real-trained probe on real attack runs (content call_ends); "
             f"{d['impactful calls']} impactful calls in {d['attack runs with an impactful call']} runs.\nCaught: score above a threshold that flags "
             f"{d['budget']:.0%} of read-only calls, set per fold on that fold's own read-only calls (in-sample); a run is caught if any impactful "
             f"call is flagged.\nShift: lower impactful calls' scores by a constant. Dilution: replace a seeded random share of impactful calls with "
             f"read-only scores. Source: {path.name} (#248).\nNot comparable to figure 2: these are the fixed-probe 2x2's scores (content call_ends, one fit), so the baseline pAUC is "
             f"{b['pauc']:.3f},\nnot the refit's 0.969 (refit_pauc_2026-10-08_233032.json: matched calls, layer and C re-chosen per fold); catch here is at "
             f"{d['budget']:.0%} false alarms\nwith an in-sample threshold, not figure 2's 1% on a held-out fold. No confidence intervals.", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    out = HERE / "results" / f"pauc_meaning_table_{path.stem.removeprefix('pauc_meaning_')}_v2.png"
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
