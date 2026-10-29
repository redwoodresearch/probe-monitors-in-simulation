"""Q3.2 figure (post hoc): the five deception-probe effects with 95% intervals against the ±0.25-SD margin (family A,
label separation; family B, real vs simulated, with real-vs-real beside each); beside it, the plant sweep.

  uv run --no-project --with matplotlib python experiments/2026-10-11_q2_margins_POSTHOC/plot_q32_POSTHOC.py <q32 result json>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
SHORT = {"1": "A1 sabotage\nlabels", "2": "A2 false vs\ntrue 'done'", "3": "B3 'done' claims\nsim (Sept) − real",
         "4": "B4 'done' claims\nsim (today) − real", "5": "B5 run-level\nsim − real"}
YMAX, YMIN = 1.0, -0.8


def bar(ax, x, p, lo, hi, col, label=None):
    ax.errorbar(x, p, yerr=[[p - max(lo, YMIN)], [min(hi, YMAX) - p]], fmt="o", color=col, capsize=4, label=label)
    if hi > YMAX:
        ax.annotate(f"to {hi:+.2f}", (x, YMAX), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color=col)
    ax.annotate(f"{p:+.2f}", (x, p), textcoords="offset points", xytext=(5, 0), ha="left", va="center", fontsize=8, color=col)


def main(path: Path):
    d = json.loads(path.read_text()); keys = [k for k in d if k[:1] in SHORT]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, k in enumerate(keys):
        o = d[k]; first = i == 0
        bar(a, i - 0.12, o["point"], *o["95%"], "C0", "effect" if first else None)
        rvr = next((v for kk, v in o.items() if kk.startswith("real-vs-real")), None)
        if rvr:
            bar(a, i + 0.12, rvr["point"], *rvr["95%"], "0.55", "real-vs-real" if k[0] == "3" else None)
    for sgn in (1, -1):
        a.axhline(0.25 * sgn, color="C3", ls="--", lw=1, label="±0.25 SD margin" if sgn == 1 else None)
    a.axhline(0, color="k", lw=0.5); a.axvline(1.5, color="0.8", lw=1)
    a.set_xticks(range(len(keys)), [SHORT[k[0]] for k in keys], fontsize=8); a.set_ylim(YMIN, YMAX)
    a.set_ylabel("effect, SD units (closer to 0 = smaller effect)"); a.set_title("A: label separation  |  B: real vs simulated (95% intervals)")
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    for k in keys:
        sw = d[k]["plant sweep (detection rate per size in SD)"]
        b.plot([float(s) for s in sw], list(sw.values()), marker="o", ms=3, label=SHORT[k[0]].replace("\n", " "))
    b.axhline(0.8, color="0.5", ls="--", lw=1); b.set_xlabel("planted shift, SD"); b.set_ylabel("share of 200 copies detected")
    b.set_title("Plant sweep: sensitivity on fixed data, not power"); b.legend(fontsize=7)
    fig.tight_layout()
    dest = HERE / "figures" / f"{path.stem}.png"; dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest, dpi=110); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
