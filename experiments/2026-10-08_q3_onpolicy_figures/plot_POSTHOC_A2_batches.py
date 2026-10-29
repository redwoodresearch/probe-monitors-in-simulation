"""A2 figure (post hoc framework, C prospective): left, the three batch pairs' difference in runs with any alarm in the
first 20 read-only calls, with 95% intervals (thresholds re-estimated) and the +-10 band; right, runs with >= 50 alarms
per batch, split into stuck and not stuck. $0.

  uv run --no-project --with matplotlib python plot_POSTHOC_A2_batches.py <A2_batches_POSTHOC_*.json>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
NAME = {"honA": "A", "honB": "B", "honC": "C"}


def main(path: str):
    R = json.loads(Path(path).read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    pairs = list(R["pairs"].items())
    for y, (k, v) in enumerate(pairs[::-1]):
        a, b = k.split("-")
        if v is None:  # an empty three-way set: no estimate
            a1.text(0, y, f"{NAME[a]} − {NAME[b]}: unavailable ({'; '.join(R['category_reasons'].get(k, []))})", ha="center", fontsize=9)
            continue
        label = f"{NAME[a]} − {NAME[b]} ({R['categories'][k]})\n{100 * v['share'][a]:.1f}% vs {100 * v['share'][b]:.1f}%"
        iv, d = v["ci95_thresholds_reestimated"], v["difference"]
        if iv:  # a percentile interval need not contain its point, so it is drawn on its own
            a1.hlines(y, 100 * iv[0], 100 * iv[1], color="#4C72B0" if "honC" in k else "gray", lw=2)
            a1.vlines([100 * iv[0], 100 * iv[1]], y - 0.12, y + 0.12, color="#4C72B0" if "honC" in k else "gray", lw=2)
        a1.plot(100 * d, y, "o", color="#4C72B0" if "honC" in k else "gray", ms=8)
        a1.annotate(f"{100 * d:+.1f} [{100 * iv[0]:+.1f}, {100 * iv[1]:+.1f}]" if iv else f"{100 * d:+.1f} (not reported)",
                    (100 * d, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10)
        a1.text(1.01, y, label, transform=a1.get_yaxis_transform(), va="center", fontsize=8)
    a1.axvspan(-10, 10, color="#55A868", alpha=0.08, label="±10-point margin")
    a1.axvline(0, color="black", lw=0.8)
    a1.set_yticks([])
    a1.set_ylim(-0.6, len(pairs) - 0.3)
    a1.set_xlabel("difference in runs with any alarm, first 20 read-only calls\n(percentage points; 95% intervals; 0 = agree)")
    a1.legend(fontsize=8, loc="lower left")
    a1.set_title(f"Batch reading: {R['batch_reading']}", fontsize=10)
    H = R["alarm_heavy"]
    xs = list(NAME)
    stuck = [H[w]["stuck_and_ge50"] for w in xs]
    unknown = [sum(r["alarm_heavy"] and not r["assessable"] for r in H[w]["runs"]) for w in xs]
    other = [H[w]["runs_ge50"] - s - u for w, s, u in zip(xs, stuck, unknown)]
    names = [f"{NAME[w]}\n({H[w]['roster']} runs)" for w in xs]
    a2.bar(names, stuck, color="#C44E52", label="stuck")
    a2.bar(names, other, bottom=stuck, color="#DD8452", label="not stuck")
    a2.bar(names, unknown, bottom=[s + o for s, o in zip(stuck, other)], color="#BBBBBB", label="not assessable")
    for i, w in enumerate(xs):
        a2.annotate(str(H[w]["runs_ge50"]), (i, H[w]["runs_ge50"]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10)
    a2.set_ylabel("runs with ≥ 50 alarms")
    a2.yaxis.get_major_locator().set_params(integer=True)
    a2.set_ylim(0, max(3, max(H[w]["runs_ge50"] for w in xs) + 1))
    a2.legend(fontsize=8)
    a2.set_title(f"Stuck, alarm-heavy runs: {R['stuck_reading']}", fontsize=10)
    fig.tight_layout()
    dst = HERE / "figures" / f"{Path(path).stem}.png"
    dst.parent.mkdir(exist_ok=True)
    fig.savefig(dst, dpi=150)
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    main(ap.parse_args().path)
