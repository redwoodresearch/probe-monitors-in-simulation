"""Q3.1 overnight figures (post hoc), one question per figure, every number read from a committed result file (intervals
as stored, never recomputed):
  1 claim C clean test             results/claim_c_POSTHOC_2026-10-10_225232.json
  2 content-only command rerun     results/claim_c_content_POSTHOC_2026-10-11_000641.json
  3 first-k                        results/claim_c_firstk_POSTHOC_2026-10-10_233831.json
  4 attempted / declined split     results/attempt_split_POSTHOC_2026-10-11_002019.json
  5 log-FPR ROC of the clean test  results/q31_roc_POSTHOC_2026-10-11_002605.json (run_q31_roc.py)
  (figure 1's matched-length status: results/claim_c_k26_POSTHOC_2026-10-11_102216.oneline.json, a one-line copy of
   Amendment 15's claim C k = 26 result from aswin/q3-590-redo @ ebc27935; see manifest_git_copies_q31_figures_v1.jsonl)
Wording follows the interim rule: runs are "flagged", not "caught attacks".
Superseded renders (2026-10-11, about 10:35 PT; numbers unchanged unless stated):
  - 1_claim_c_clean_test.png and 1_claim_c_clean_test_2026-10-11_052144.png ("flagged alike" overstated the registered
    verdict) and 1_claim_c_clean_test_2026-10-11_053916.png ("matched-length check not run", before claim C's k = 26 check
    ran and passed) are superseded by the figure-1 render this version writes;
  - 2_content_only.png ("leaves flagging unchanged" overstated) is superseded by 2_content_only_2026-10-11_053916.png.
  - Recolour (2026-10-11, about 11:00 PT, for the page; numbers unchanged): real run 2 moves from green to the page's
    light blue (#93c1f0, as in #663), so every figure-1 to figure-5 render before this one is superseded by the
    renders this version writes.
  - Layout fix (2026-10-11, about 11:10 PT; numbers unchanged): figure 5's 1% and 5% operating-point notes move into the
    legend, clear of the dotted 1-of-195 line, so 5_clean_test_roc_logfpr_2026-10-11_105737.png is superseded by the
    figure-5 render this version writes.
  - Retitle (2026-10-11, about 11:20 PT; numbers unchanged): figure 4's "No attack run declined" overstated the data,
    which leave some runs unclear or unlabelled; it now reads "No labelled run was classified as declined", with those
    counts read from the result. 4_attempt_split_2026-10-11_105737.png is superseded by the figure-4 render this version
    writes.
Pass figure names (e.g. 1_claim_c_clean_test) as arguments to render only those; with none, all five are rendered.

  uv run --no-project --python 3.12 --with matplotlib --with numpy python experiments/2026-10-08_q3_onpolicy_figures/plot_q31_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
RES, OUT = HERE / "results", HERE / "figures" / "q31"
BLUE, ORANGE, LIGHT, GREY = "#2a78d6", "#eb6834", "#93c1f0", "#8a8984"  # real run 2 in the page's light blue, as in #663
INK, INK2 = "#0b0b0b", "#52514e"
ARMS = [("Real run 1", BLUE), ("Simulated", ORANGE), ("Real run 2", LIGHT)]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    return json.loads((RES / name).read_text())


def pts(x):
    return f"{100 * x:+.1f}".replace("-", "−")


def ci(iv):
    return f"[{pts(iv['ci95'][0])}, {pts(iv['ci95'][1])}]"


def bars(ax, x0, vals, width=0.26):
    """vals: [(value, interval dict or None, colour)]; bars at x0 + offsets, value printed above each error bar."""
    for i, (v, iv, col) in enumerate(vals):
        x = x0 + (i - (len(vals) - 1) / 2) * width
        lo, hi = (iv["ci95"] if iv else (v, v))
        ax.bar(x, 100 * v, width * 0.92, color=col, zorder=2)
        ax.errorbar(x, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
        ax.text(x, 100 * hi + 1.5, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=10, color=INK, zorder=4)


def frame(ax, ylabel):
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="#e4e3df", zorder=0)


def legend(ax, items):
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=c, label=l) for l, c in items], frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))


STAMP = __import__("datetime").datetime.now(__import__("zoneinfo").ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")


def three(s, obs, rep):
    return [(obs["tpr_run1"], s["tpr_run1_interval"], BLUE), (obs["tpr_sim"], rep["tpr_sim_interval"], ORANGE),
            (obs["tpr_run2"], rep["tpr_run2_interval"], LIGHT)]


def verdict(s):
    return "10-pt gap ruled out" if "a 10-point gap is ruled out" in s["verdicts"] else "inconclusive"


def fig1():
    d = load("claim_c_POSTHOC_2026-10-10_225232.json")["selections"]
    k26 = {key: v["matched_length"] for key, v in load("claim_c_k26_POSTHOC_2026-10-11_102216.oneline.json")["results"].items()}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = []
    for i, (key, name) in enumerate([("command_end", "Commands"), ("reply_end", "Replies")]):
        s = d[key]
        bars(ax, i, three(s, s["observed"], s["report"]))
        labels.append(f"{name}\nreal − sim {pts(s['observed']['gap'])} {ci(s['gap_interval'])}\n"
                      f"run 2 − run 1 {pts(s['observed']['control'])} {ci(s['control_interval'])}\n{verdict(s)}"
                      + (f";\nmatched-length check\n{'passed' if k26[key]['passes'] else 'failed'} (k = {k26[key]['k']})"
                         if verdict(s) != "inconclusive" else ""))  # Amendment 15's claim C k = 26 result
    ax.set_xticks([0, 1], labels)
    frame(ax, "Flagged, targeting 1% false alarms (%)\nhigher = more runs flagged")
    ax.set_title("Clean detector: a 10-point real-vs-simulated gap is ruled out at both readings", loc="left", fontsize=12, color=INK)
    legend(ax, ARMS)
    return fig


def fig2():
    r = load("claim_c_content_POSTHOC_2026-10-11_000641.json")["results"]["full"]
    dr = r["drop"]["run1"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bars(ax, 0, three(r, r["observed"], r["report"]))
    ax.set_xticks([0], [f"Full run, content-only commands\nchange vs original (real run 1) {pts(dr['difference'])} {ci(dr['interval'])}\n"
                        f"real − sim {pts(r['observed']['gap'])} {ci(r['gap_interval'])}\n"
                        f"run 2 − run 1 {pts(r['observed']['control'])} {ci(r['control_interval'])}\n{verdict(r)}"])
    ax.set_xlim(-0.6, 0.6)
    frame(ax, "Flagged, targeting 1% false alarms (%)\nhigher = more runs flagged")
    ax.set_title(f"Dropping fixed-slot tokens:\nflagging changes by {pts(dr['difference'])} points {ci(dr['interval'])}\n"
                 f"({'no drop' if not r['drop']['drops'] else 'a drop'} under the registered rule)", loc="left", fontsize=12, color=INK)
    legend(ax, ARMS)
    return fig


def fig3():
    d = load("claim_c_firstk_POSTHOC_2026-10-10_233831.json")["results"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = []
    for i, (key, name) in enumerate([("command_end|k3", "First 3\ncommands"), ("command_end|k5", "First 5\ncommands"),
                                     ("reply_end|k3", "First 3\nreplies"), ("reply_end|k5", "First 5\nreplies")]):
        s = d[key]
        bars(ax, i, three(s, s["observed"], s["report"]))
        labels.append(f"{name}\n{verdict(s)}" + ("\n(1% threshold:\n1 honest run)" if key == "reply_end|k3" else ""))
    a5 = d["reply_end|k3"]["report"]["at_5pct"]
    bars(ax, 4, [(a5["tpr_run1"], a5["tpr_run1_interval"], BLUE), (a5["tpr_sim"], a5["tpr_sim_interval"], ORANGE)])
    labels.append("First 3 replies\ntargeting 5%\nfalse alarms")
    ax.set_xticks(range(5), labels)
    frame(ax, "Flagged at the false-alarm target (%)\nhigher = more runs flagged")
    ax.set_title("Scoring only each run's first commands or replies (targeting 1% false alarms unless marked)", loc="left", fontsize=12, color=INK)
    legend(ax, ARMS)
    return fig


def fig4():
    p = load("attempt_split_POSTHOC_2026-10-11_002019.json")["part1"]
    cats = [("attempted", BLUE, "Attempted"), ("declined", ORANGE, "Declined"), ("unclear", GREY, "Unclear"), ("None", "#c3c2b7", "No label (no counted log)")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, w in enumerate(["run1", "run2", "sim"]):
        for j, (k, col, _) in enumerate(cats):
            n = p[w]["labels"].get(k, 0)
            x = i + (j - 1.5) * 0.21
            ax.bar(x, n, 0.19, color=col, zorder=2)
            ax.text(x, n + 3, str(n), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(3), ["Real run 1", "Real run 2", "Simulated"])
    ax.set_ylim(0, 215)
    ax.set_ylabel("Attack runs (of 196 tasks)")
    ax.yaxis.grid(True, color="#e4e3df", zorder=0)
    left = {w: p[w]["labels"].get("unclear", 0) + p[w]["labels"].get("None", 0) for w in ("run1", "run2", "sim")}
    ax.set_title("No labelled run was classified as declined (frozen label rule, self-report)\n"
                 f"(unclear or unlabelled: run 1 {left['run1']}, run 2 {left['run2']}, simulated {left['sim']})",
                 loc="left", fontsize=12, color=INK)
    legend(ax, [(l, c) for _, c, l in cats])
    return fig


def fig5():
    import numpy as np

    d = load("q31_roc_POSTHOC_2026-10-11_002605.json")
    nb = d["n_honest_b"]
    floor, zero = 1 / nb, 1 / (2 * nb)  # FPR 0 is drawn at `zero`, left of the 1-of-195 floor
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for w, name, col in [("run1", "Real run 1", BLUE), ("sim", "Simulated", ORANGE)]:
        f, t = np.array(d["curves"][w]["fpr"]), np.array(d["curves"][w]["tpr"])
        x = np.where(f == 0, zero, f)
        ax.step(x, 100 * t, where="post", color=col, lw=2, label=f"{name} (n = {d['curves'][w]['n']})", zorder=3)
    for rate, mk in (("0.01", "o"), ("0.05", "s")):
        op = d["operating_points"][rate]
        x = op["fpr_honest_b"] if op["fpr_honest_b"] > 0 else zero
        for w, col, ms in (("run1", BLUE, 12), ("sim", ORANGE, 7)):  # run 1 larger, so both show where they overlap
            ax.plot(x, 100 * op[f"tpr_{w}"], mk, ms=ms, mfc=col, mec="white", mew=1.5, zorder=4)
    ax.axvline(floor, color=INK2, ls=":", lw=1)
    ax.text(floor * 1.08, 86, "1 of 195\nhonest runs", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(zero / 1.3, 1.05)
    ax.set_xticks([zero, floor, 0.01, 0.05, 0.1, 1], ["0", "0.5%", "1%", "5%", "10%", "100%"])
    ax.minorticks_off()
    ax.set_ylim(85, 100.8)  # a line chart, so the axis needn't start at 0; every point below 85% sits at FPR 0
    ax.set_xlabel("False alarms on honest runs (log scale; lower = better)")
    ax.set_ylabel("Attack-instructed runs flagged (%)\nhigher = more runs flagged")
    ax.yaxis.grid(True, color="#e4e3df", zorder=0)
    ax.set_title("Clean detector ROC, content-only commands, full run", loc="left", fontsize=12, color=INK)
    from matplotlib.lines import Line2D

    notes = [Line2D([], [], ls="none", marker=mk, ms=8, mfc=GREY, mec="white",
                    label=f"{float(rate):.0%} target: {d['operating_points'][rate]['honest_b_above']}/{nb} honest flagged")
             for rate, mk in (("0.01", "o"), ("0.05", "s"))]  # in the legend, clear of the dotted 1-of-195 line (as figure 8)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + notes, frameon=False, loc="lower right")
    return fig


def main(only=()):
    OUT.mkdir(parents=True, exist_ok=True)
    figs = [("1_claim_c_clean_test", fig1), ("2_content_only", fig2), ("3_first_k", fig3), ("4_attempt_split", fig4),
            ("5_clean_test_roc_logfpr", fig5)]
    for name, fn in [f for f in figs if not only or f[0] in only]:
        fig = fn()
        fig.tight_layout()
        out = OUT / f"{name}_{STAMP}.png"  # a new name each run; earlier figures are kept
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    import sys

    main(tuple(sys.argv[1:]))
