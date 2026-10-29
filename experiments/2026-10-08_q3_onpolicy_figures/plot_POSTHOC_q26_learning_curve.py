"""Q2.6 learning-curve figure (post hoc): score against training tasks per fold model, log x-axis. $0, local.

Left: AUROC on real runs (each seed a dot, the median a line; the 20-task pilot's 0.975 and the full-data value dashed).
Right: catch at the frozen 1% and 5% thresholds (medians with the seed range), with the median achieved false-alarm rates
printed. Training tasks per fold model: 3n/5 for the subsamples, the mean over models for the full data.

  uv run --no-project --with numpy --with matplotlib python plot_POSTHOC_q26_learning_curve.py <learning_curve_POSTHOC_q26_*.json> <per-call draw-0 file>
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

HERE = Path(__file__).parent
PILOT = 0.975  # Qwen acting, 20 tasks (figure 3), notes_offpolicy_checks_2026-10-09.md


def main(path: str, percall: str):
    R = json.loads(Path(path).read_text())
    if not R["gate"]["passed"]:
        raise SystemExit("the gate failed; nothing to plot")
    fold_n = Counter(json.loads(Path(percall).read_text())["folds"].values())
    full_x = sum(sum(fold_n[f] for f in range(5) if f not in (k, (k + 1) % 5)) for k in range(5)) / 5
    by = {}
    for r in R["configs"]:
        m = r["metrics"]["real->real"]
        x = full_x if r["n"] is None else 3 * r["n"] / 5
        by.setdefault(x, []).append(m)
    xs = sorted(by)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for x in xs:
        v = [m["auroc"] for m in by[x] if m["auroc"] is not None]
        a1.scatter([x] * len(v), v, s=14, color="#4C72B0", alpha=0.6)
    med = [median([m["auroc"] for m in by[x] if m["auroc"] is not None]) for x in xs]
    a1.plot(xs, med, "-o", color="#4C72B0", label="median AUROC (real runs)")
    for x, y in zip(xs, med):
        a1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    a1.axhline(PILOT, ls="--", color="#C44E52", label="20-task pilot (0.975; different held-out tasks)")
    a1.axhline(med[-1], ls=":", color="gray", label=f"full data ({med[-1]:.3f})")
    a1.set_xscale("log"); a1.set_xticks(xs); a1.set_xticklabels([f"{x:.0f}" for x in xs]); a1.xaxis.set_minor_formatter(NullFormatter())
    a1.set_xlabel("training tasks per fold model (log)"); a1.set_ylabel("AUROC on real runs\n(higher is better)"); a1.legend(fontsize=8)
    for key, fa, color, lab in (("catch_frozen", "false_alarm_frozen", "#55A868", "1% target"), ("catch_frozen_5pct", "false_alarm_frozen_5pct", "#8172B2", "5% target")):
        mid, lo, hi, fam = [], [], [], []
        for x in xs:
            v = [m[key] for m in by[x] if m.get(key) is not None]
            mid.append(100 * median(v)); lo.append(100 * min(v)); hi.append(100 * max(v))
            fam.append(100 * median([m[fa] for m in by[x] if m.get(fa) is not None]))
        a2.errorbar(xs, mid, yerr=[[m - l for m, l in zip(mid, lo)], [h - m for m, h in zip(mid, hi)]], fmt="-o", color=color, capsize=3, label=f"{lab} (median; seed range)")
        for x, y, f in zip(xs, mid, fam):
            a2.annotate(f"{y:.1f}% (FA {f:.1f}%)", (x, y), textcoords="offset points", xytext=(30, -4) if lab == "1% target" else (0, 8), ha="left" if lab == "1% target" else "center", fontsize=8)
    a2.set_xscale("log"); a2.set_xticks(xs); a2.set_xticklabels([f"{x:.0f}" for x in xs]); a2.xaxis.set_minor_formatter(NullFormatter())
    a2.set_xlabel("training tasks per fold model (log)"); a2.set_ylabel("impactful commands caught (%)\n(higher is better)"); a2.legend(fontsize=8)
    fig.suptitle(f"Q2.6: the monitor's score against training size (post hoc; reading: {R['reading']['reading']})", fontsize=10)
    fig.tight_layout()
    dst = HERE / "figures" / f"{Path(path).stem}.png"
    dst.parent.mkdir(exist_ok=True)
    fig.savefig(dst, dpi=150)
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("percall")
    a = ap.parse_args()
    main(a.path, a.percall)
