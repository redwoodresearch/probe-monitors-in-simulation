"""Write-up figure 555eb7a0, "Whole-run attack detector, real vs simulated", regenerated from raw data ($0, local).

Source: experiments/2026-10-08_q3_onpolicy_figures/results/claim_c_POSTHOC_2026-10-10_225232.json (committed at
ab6facfe, identical on main; sha256 checked below). Claim C's whole-run attack detector, scored at its 1% false-alarm
threshold (set on honest_a, achieved rate on honest_b), on the 183-task cohort:
  "gap"     = catch on real run 1 minus catch on simulated runs  (real - sim)
  "control" = catch on real run 2 minus catch on real run 1      (two real runs; the label in plot_q31_stripped.py)
for two readouts: commands (selections.command_end) and replies (selections.reply_end). Intervals: 95%, tasks resampled
with the threshold recomputed per draw (the file's gap_interval and control_interval).

Every drawn value is read by key, printed with its key path, and compared with the number printed on the doc image
(FIGURES.md @ af47bc10, row 555eb7a0), rounded to one decimal in points: MATCH or MISMATCH.

  uv run --no-project --with matplotlib --with numpy python experiments/2026-10-12_q3_writeup_figures/plot_fig_555eb7a0.py \
      experiments/2026-10-08_q3_onpolicy_figures/results/claim_c_POSTHOC_2026-10-10_225232.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SHA256 = "6a4c5281b7a24dd01bc687fa4612676415d8324999fb85fdb1d5a9683eb37bf3"  # claim_c_POSTHOC_2026-10-10_225232.json @ ab6facfe
# (key path, point key, interval key) -> the doc image's printed number (points, [lo, hi])
DOC = {
    ("command_end", "gap"): (1.6, (-1.1, 5.3)),
    ("command_end", "control"): (1.6, (0.0, 4.9)),
    ("reply_end", "gap"): (2.7, (-1.6, 7.6)),
    ("reply_end", "control"): (1.1, (-2.3, 4.9)),
}
LABEL = {"gap": "real − simulated", "control": "two real runs\n(run 2 − run 1)"}
READOUT = {"command_end": "Commands", "reply_end": "Replies"}
COLOR = {"gap": "#2a6fb0", "control": "#9a9a9a"}


def main(src):
    data = Path(src).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        raise SystemExit(f"{src}: sha256 {got} is not the committed file's {SHA256}")
    d = json.loads(data)
    rows, ok = [], True
    for (sel, kind), (doc_pt, doc_ci) in DOC.items():
        s = d["selections"][sel]
        pt = 100 * s["observed"][kind]
        lo, hi = (100 * v for v in s[f"{kind}_interval"]["ci95"])
        if s[f"{kind}_interval"]["failed"] or s[f"{kind}_interval"]["invalid"]:
            raise SystemExit(f"selections.{sel}.{kind}_interval has failed or invalid draws")
        mine = (round(pt, 1), (round(lo, 1), round(hi, 1)))
        match = mine[0] == doc_pt and mine[1] == tuple(doc_ci)
        ok &= match
        print(f"selections.{sel}.observed.{kind} = {pt:+.3f}; selections.{sel}.{kind}_interval.ci95 = [{lo:+.3f}, {hi:+.3f}]"
              f" -> {mine[0]:+.1f} ({mine[1][0]:+.1f} to {mine[1][1]:+.1f}); doc {doc_pt:+.1f} ({doc_ci[0]:+.1f} to {doc_ci[1]:+.1f})"
              f" {'MATCH' if match else 'MISMATCH'}")
        rows.append({"selection": sel, "kind": kind, "key_point": f"selections.{sel}.observed.{kind}",
                     "key_interval": f"selections.{sel}.{kind}_interval.ci95", "point_pts": pt, "ci95_pts": [lo, hi],
                     "doc_point_pts": doc_pt, "doc_ci95_pts": list(doc_ci), "match": match})

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs, ticks = [], []
    for g, sel in enumerate(READOUT):
        for j, kind in enumerate(("gap", "control")):
            r = next(r for r in rows if r["selection"] == sel and r["kind"] == kind)
            x = g * 2.6 + j
            xs.append(x)
            ax.bar(x, r["point_pts"], width=0.8, color=COLOR[kind], zorder=2)
            ax.errorbar(x, r["point_pts"], yerr=[[r["point_pts"] - r["ci95_pts"][0]], [r["ci95_pts"][1] - r["point_pts"]]],
                        fmt="none", ecolor="#222222", capsize=5, lw=1.2, zorder=3)
            ax.text(x, r["ci95_pts"][1] + 0.6, f"{r['point_pts']:+.1f}\n({r['ci95_pts'][0]:+.1f} to {r['ci95_pts'][1]:+.1f})",
                    ha="center", va="bottom", fontsize=8.5)
        ticks.append((g * 2.6 + 0.5, READOUT[sel]))
    ax.axhline(0, color="#222222", lw=0.8, zorder=1)
    for y in (-10, 10):
        ax.axhline(y, color="#c0392b", lw=0.8, ls="--", zorder=1)
    ax.set_xticks([t for t, _ in ticks], [n for _, n in ticks])
    ax.set_ylim(-12, 14)
    ax.set_ylabel("Difference in runs caught at a 1% false-alarm target\n(percentage points; closer to 0 = no shift)")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=COLOR["gap"], label=LABEL["gap"]), Patch(color=COLOR["control"], label=LABEL["control"]),
                       Line2D([0], [0], color="#c0392b", ls="--", lw=0.8, label="±10-point margin")],
              loc="lower right", fontsize=8, frameon=False,
              title="whiskers and parentheses: 95% CI", title_fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Whole-run attack detector, real vs simulated", loc="left", fontsize=11)
    # achieved false alarms on the held-out honest_b runs at each readout's 1% threshold, read by key
    fa = {}
    for sel in READOUT:
        s = d["selections"][sel]
        lo, hi = (100 * v for v in s["false_alarm_interval"]["ci95"])
        fa[sel] = {"key_point": f"selections.{sel}.observed.false_alarm_honest_b", "key_interval": f"selections.{sel}.false_alarm_interval.ci95",
                   "point_pct": 100 * s["observed"]["false_alarm_honest_b"], "ci95_pct": [lo, hi]}
        print(f"{fa[sel]['key_point']} = {fa[sel]['point_pct']:.3f}%; {fa[sel]['key_interval']} = [{lo:.3f}, {hi:.3f}]%")
    fig.text(0.01, 0.01, "Achieved false alarms on held-out honest runs: " + "; ".join(
        f"{READOUT[k].lower()} {v['point_pct']:.1f}% ({v['ci95_pct'][0]:.1f} to {v['ci95_pct'][1]:.1f})" for k, v in fa.items()),
        fontsize=7.5, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    png = HERE / "figures" / f"fig_555eb7a0_whole_run_detector_{stamp}.png"
    with png.open("xb") as fh:  # never overwrite a rendered figure
        fig.savefig(fh, dpi=180, format="png")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"figure": "555eb7a0", "title": "Whole-run attack detector, real vs simulated", "git": head,
           "source": {"path": str(src), "sha256": got, "committed_at": "ab6facfe"}, "all_match": ok, "rows": rows, "achieved_false_alarms": fa,
           "png": str(png.relative_to(HERE.parent.parent)), "written": stamp}
    with (HERE / "results" / f"fig_555eb7a0_values_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(f"all_match {ok}; wrote {png} and results/fig_555eb7a0_values_{stamp}.json")


if __name__ == "__main__":
    main(sys.argv[1])
