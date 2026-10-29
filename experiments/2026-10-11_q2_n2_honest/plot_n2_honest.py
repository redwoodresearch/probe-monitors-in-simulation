"""N2 honest figure: each measure's real-minus-simulated and real-run-1-minus-run-2 gaps with 95% intervals, divided by
the measure's margin (so the margin is ±1); beside it, the plant sweep with plant size in margin units.

  uv run --no-project --with matplotlib python experiments/2026-10-11_q2_n2_honest/plot_n2_honest.py <n2 result json>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
UNIT = {"logpath_share": (100, " pts"), "readbacks": (1, " /run")}
NAMES = {"logpath_share": "log/audit/history-path share\n(margin ±1 point)", "readbacks": "read-backs after writing\n(margin ±0.5 per run)"}


def main(path: Path):
    d = json.loads(path.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    for i, (m, name) in enumerate(NAMES.items()):
        o = d[m]; mg = o["margin"]
        for off, part, col, lab in ((-0.12, "real minus sim", "C0", "real − simulated"), (0.12, "real run 1 minus run 2", "0.55", "real run 1 − run 2")):
            p, (lo, hi) = o[part]["point"], o[part]["95%"]
            a.errorbar(i + off, p / mg, yerr=[[(p - lo) / mg], [(hi - p) / mg]], fmt="o", color=col, capsize=4, label=lab if i == 0 else None)
            k, u = UNIT[m]
            a.annotate(f"{p * k:+.2f}{u}", (i + off, p / mg), textcoords="offset points", xytext=(5, 0), ha="left", va="center", fontsize=8, color=col)
        a.annotate(o["verdict"], (i, 1.35), ha="center", fontsize=9)
    for sgn in (1, -1):
        a.axhline(sgn, color="C3", ls="--", lw=1, label="margin" if sgn == 1 else None)
    a.axhline(0, color="k", lw=0.5); a.set_ylim(-2.2, 1.6)
    a.set_xticks(range(len(NAMES)), list(NAMES.values()), fontsize=8)
    a.set_ylabel("gap ÷ margin (inside ±1 = within margin)"); a.set_title(f"Sol's honest runs, {d['tasks, primary (all three runs, both real runs overlap800)']} tasks: real vs simulated (95% intervals)")
    a.legend(fontsize=8, loc="lower left")
    for m, name in NAMES.items():
        sw = d[m]["plant sweep (detection rate per size)"]; mg = d[m]["margin"]
        b.plot([float(s) / mg for s in sw], list(sw.values()), marker="o", ms=3, label=name.split("\n")[0])
    b.axhline(0.8, color="0.5", ls="--", lw=1); b.set_xlabel("planted shift ÷ margin"); b.set_ylabel("share of 200 copies detected")
    b.set_title("Plant sweep: sensitivity on fixed data, not power"); b.legend(fontsize=8)
    fig.tight_layout()
    dest = HERE / "figures" / f"{path.stem}.png"; dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest, dpi=110); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
