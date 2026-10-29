"""Q2.5 figure (post hoc): each real-minus-simulated catch gap and its real-vs-real control with 95% intervals against
the ±5 and ±10-point margins; beside it, the plant sweep's detection rate per size.

  uv run --no-project --with matplotlib python experiments/2026-10-11_q2_margins_POSTHOC/plot_q25_POSTHOC.py <q25 result json>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
LABELS = {"destructive_raw target 1%": "Destructive, 1%", "destructive_raw target 5%": "Destructive, 5%",
          "transfer target 1%": "Transfer, 1%", "transfer target 5%": "Transfer, 5%"}


def main(path: Path):
    d = json.loads(path.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, (k, lab) in enumerate(LABELS.items()):
        for off, part, col, name in ((-0.12, "gap", "C0", "real − simulated"), (0.12, "real-vs-real (run 1 minus run 2)", "0.55", "real run 1 − run 2")):
            o = d[k][part]; p, lo, hi = 100 * o["point"], 100 * o["95%"][0], 100 * o["95%"][1]
            a.errorbar(i + off, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col, capsize=4, label=name if i == 0 else None)
            a.annotate(f"{p:+.1f}", (i + off, hi), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color=col)
    for m, ls in ((5, ":"), (10, "--")):
        for sgn in (1, -1):
            a.axhline(sgn * m, color="C3", ls=ls, lw=1, label=f"±{m} margin" if sgn == 1 else None)
    a.axhline(0, color="k", lw=0.5)
    a.set_xticks(range(4), list(LABELS.values())); a.set_ylabel("catch-rate gap, points (closer to 0 is better)")
    a.set_title("Gaps with 95% intervals (thresholds re-estimated)"); a.set_ylim(-14, 15)
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    for k, lab in LABELS.items():
        sw = d[k]["plant sweep (detection rate per size in points)"]
        b.plot([int(s) for s in sw], list(sw.values()), marker="o", ms=3, label=lab)
    b.axhline(0.8, color="0.5", ls="--", lw=1); b.set_xlabel("planted gap, points"); b.set_ylabel("share of 200 copies detected")
    b.set_title("Plant sweep: sensitivity on fixed data, not power"); b.legend(fontsize=8)
    fig.tight_layout()
    dest = HERE / "figures" / f"{path.stem}.png"; dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest, dpi=110); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
