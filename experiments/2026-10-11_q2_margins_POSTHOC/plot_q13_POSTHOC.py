"""Q1.3 figure (post hoc): each behaviour gap (run 1 minus sim) and its real-vs-real (run 1 minus run 2), divided by the
measure's set-1 margin so the margin is ±1 for every measure; set 2 (steps, failure text) marked; beside it, the plant
sweep with plant size in margin units. Raw values are printed on the points.

  uv run --no-project --with matplotlib python experiments/2026-10-11_q2_margins_POSTHOC/plot_q13_POSTHOC.py <q13 result json>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
NAMES = {"sh_edit": "edit share", "reread_rate": "re-read rate", "sh_failtext": "failure-text share", "n_steps": "steps", "tok_out": "output tokens"}


def main(path: Path):
    d = json.loads(path.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    for i, (m, name) in enumerate(NAMES.items()):
        o = d[m]; mg = o["margin set 1"]
        for off, part, col, lab in ((-0.12, "run1 - sim", "C0", "real run 1 − sim"), (0.12, "run1 - run2", "0.55", "real run 1 − run 2")):
            p, (lo, hi) = o[part]["point"], o[part]["95%"]
            a.errorbar(i + off, p / mg, yerr=[[(p - lo) / mg], [(hi - p) / mg]], fmt="o", color=col, capsize=4, label=lab if i == 0 else None)
            a.annotate(f"{p:+,.0f}" if abs(p) >= 100 else f"{p:+.3g}", (i + off, p / mg), textcoords="offset points", xytext=(5, 0), ha="left", va="center", fontsize=7, color=col)
        if o["margin set 2"]:
            r = o["margin set 2"] / mg
            a.plot([i - 0.35, i + 0.35], [r, r], color="C1", lw=1.5); a.plot([i - 0.35, i + 0.35], [-r, -r], color="C1", lw=1.5,
                                                                           label="set 2 (Aswin's resolution)" if m == "sh_failtext" else None)
    for sgn in (1, -1):
        a.axhline(sgn, color="C3", ls="--", lw=1, label="set 1 margin (forecast width)" if sgn == 1 else None)
    a.axhline(0, color="k", lw=0.5)
    a.set_xticks(range(len(NAMES)), [f"{n}\n±{d[m]['margin set 1']:,.0f}" if d[m]["margin set 1"] >= 100 else f"{n}\n±{d[m]['margin set 1']:.3g}" for m, n in NAMES.items()], fontsize=8)
    a.set_ylabel("gap ÷ set-1 margin (inside ±1 = within margin)"); a.set_title("Behaviour gaps with 95% intervals, 196 tasks")
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4)
    for m, name in NAMES.items():
        sw = d[m]["plant sweep (detection rate per size, run 1 - sim)"]; mg = d[m]["margin set 1"]
        b.plot([float(s) / mg for s in sw], list(sw.values()), marker="o", ms=3, label=name)
    b.axhline(0.8, color="0.5", ls="--", lw=1); b.set_xlabel("planted shift ÷ set-1 margin"); b.set_ylabel("share of 200 copies detected")
    b.set_title("Plant sweep: sensitivity on fixed data, not power"); b.legend(fontsize=7)
    fig.tight_layout()
    dest = HERE / "figures" / f"{path.stem}.png"; dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest, dpi=110); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
