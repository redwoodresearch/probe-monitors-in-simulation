"""Q3.1 figures for the stripped-input test (plan Amendment 11, post hoc), in plot_q31_figures.py's style, every number
read from committed result files (intervals as stored):
  7 (7_stripped_clean_test.png) claim C's clean detector beside the same detector retrained with the agent's reasoning and chat removed: flagged
    targeting 1% false alarms, real run 1 vs simulated vs real run 2, commands and replies
      results/claim_c_POSTHOC_2026-10-10_225232.json and the stripped result file (argument)
  8 (8_stripped_roc_logfpr.png) log-FPR ROC of the stripped detector, real run 1 and simulated against honest_b, from the stored run scores; its 1%
    point must reproduce the stripped result's tpr_run1 and tpr_sim
Wording follows the interim rule: runs are "flagged", not "caught attacks".
Layout fix (2026-10-11, about 10:20 PT, from session 09's figure audit): each run now writes timestamped PNGs
(7_stripped_clean_test_<stamp>.png, 8_stripped_roc_logfpr_<stamp>.png); figure 7's values and 95% intervals sit inside
the bar bases, clear of the whiskers, and real run 2 uses the page's light blue; figure 8's operating-point notes move
into the legend, clear of the dotted 1/n line. The first renders (7_stripped_clean_test.png, 8_stripped_roc_logfpr.png)
are superseded by these; their numbers are unchanged.

  uv run --no-project --python 3.12 --with matplotlib --with numpy python \
      experiments/2026-10-08_q3_onpolicy_figures/plot_q31_stripped.py <claim_c_noagent_POSTHOC_*.json>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
RES, OUT = HERE / "results", HERE / "figures" / "q31"
BLUE, ORANGE, LIGHT, GREY = "#2a78d6", "#eb6834", "#93c1f0", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
ARMS = [("Real run 1", BLUE), ("Simulated", ORANGE), ("Real run 2", LIGHT)]
LABELS = ["phase2_attempt_labels_2026-10-09_000523_rerun_strict_selfreport_2026-10-10_111809.jsonl",
          "phase2_attempt_labels_2026-10-09_173815_scaleup_canary_selfreport_2026-10-10_111817.jsonl",
          "phase2_attempt_labels_2026-10-09_182505_scaleup_selfreport_2026-10-10_111838.jsonl",
          "phase2_attempt_labels_2026-10-10_001314_scaleup_topup_selfreport_2026-10-10_111821.jsonl",
          "phase2_attempt_labels_2026-10-10_001314_second_attack_2026-10-10_111840.jsonl"]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def pts(x):
    return f"{100 * x:+.1f}".replace("-", "−")


def ci(iv):
    return f"[{pts(iv['ci95'][0])}, {pts(iv['ci95'][1])}]"


def bars(ax, x0, vals, width=0.26):
    for i, (v, iv, col) in enumerate(vals):
        x = x0 + (i - (len(vals) - 1) / 2) * width
        lo, hi = (iv["ci95"] if iv else (v, v))
        ax.bar(x, 100 * v, width * 0.92, color=col, zorder=2)
        ax.errorbar(x, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
        fg = INK if col == LIGHT else "white"  # inside the bar base: every bar here is above 80%, every whisker above 25%
        ax.text(x, 12, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=fg, zorder=4)
        ax.text(x, 10, f"[{100 * lo:.1f},\n{100 * hi:.1f}]", ha="center", va="top", fontsize=7.5, color=fg, zorder=4)


def three(s):
    o, r = s["observed"], s["report"]
    return [(o["tpr_run1"], s["tpr_run1_interval"], BLUE), (o["tpr_sim"], r["tpr_sim_interval"], ORANGE), (o["tpr_run2"], r["tpr_run2_interval"], LIGHT)]


def verdict(s):
    return "10-pt gap ruled out" if "a 10-point gap is ruled out" in s["verdicts"] else "inconclusive"


def fig6(orig, strip):
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    labels = []
    for i, (key, name, src, tag) in enumerate([("command_end", "Commands", orig, "claim C"), ("command_end", "Commands", strip, "agent text removed"),
                                               ("reply_end", "Replies", orig, "claim C"), ("reply_end", "Replies", strip, "agent text removed")]):
        s = src[key]
        bars(ax, i, three(s))
        labels.append(f"{name}, {tag}\nreal − sim {pts(s['observed']['gap'])} {ci(s['gap_interval'])}\n"
                      f"run 2 − run 1 {pts(s['observed']['control'])} {ci(s['control_interval'])}\n{verdict(s)}")
    ax.set_xticks(range(4), labels, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Flagged, targeting 1% false alarms (%)\nhigher = more runs flagged")
    ax.yaxis.grid(True, color="#e4e3df", zorder=0)
    ax.set_title("Removing the agent's own reasoning and chat (detector retrained; off-policy ablation)", loc="left", fontsize=12, color=INK)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=c, label=l) for l, c in ARMS], frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def roc(pos, neg):
    ts = np.unique(np.concatenate([pos, neg]))[::-1]
    return [0.0] + [float((neg >= t).mean()) for t in ts], [0.0] + [float((pos >= t).mean()) for t in ts]


def fig7(strip):
    t196 = json.loads((HERE.parents[1] / "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json").read_text())
    tasks = {c["key"] for c in t196["cells"]}  # the 196 tasks (frozen_folds' keys)
    lab = {}
    for name in LABELS:
        for r in map(json.loads, (RES / name).read_text().splitlines()):
            w = {"attack": "run2" if "second_attack" in name else "run1", "sim_attack@september": "sim"}.get(r["arm"])
            if w and r.get("label"):
                lab[(w, r["task"])] = r["label"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    for ax, (key, name) in zip(axes, [("command_end", "Commands"), ("reply_end", "Replies")]):
        sc = strip[key]["scores_real_column"]
        world = np.array([u.split("|", 1)[0] for u in sc["runs"]])
        task = np.array([u.split("|", 1)[1] for u in sc["runs"]])
        z = np.array([np.nan if x is None else x for x in sc["z"]])
        have = lambda w: {t for t in task[(world == w) & np.isfinite(z)] if t in tasks}
        cohort = set.intersection(*({t for t in have(w) if lab.get((w, t)) == "attempted"} for w in ("run1", "run2", "sim")))
        pick = lambda w, ts: z[(world == w) & np.isin(task, sorted(ts)) & np.isfinite(z)]
        ha, hb = pick("honest_a", tasks), pick("honest_b", tasks)
        nb, zero = len(hb), 1 / (2 * len(hb))
        ops = {}
        for rate in (0.01, 0.05):  # q2's rule on honest_a, unit weights
            t = float(np.sort(ha)[::-1][min(int(np.floor(rate * len(ha) + 1e-9)), len(ha) - 1)])
            ops[rate] = {"fpr": float((hb > t).mean()), "above": int((hb > t).sum()), **{w: float((pick(w, cohort) > t).mean()) for w in ("run1", "sim")}}
        o = strip[key]["observed"]
        if abs(ops[0.01]["run1"] - o["tpr_run1"]) > 1e-9 or abs(ops[0.01]["sim"] - o["tpr_sim"]) > 1e-9 or len(cohort) != strip[key]["cohort_tasks"]:
            raise ValueError(f"{key}: the 1% point does not reproduce the stripped result")
        for w, lbl, col in [("run1", "Real run 1", BLUE), ("sim", "Simulated", ORANGE)]:
            f, t = map(np.array, roc(pick(w, cohort), hb))
            ax.step(np.where(f == 0, zero, f), 100 * t, where="post", color=col, lw=2, label=f"{lbl} (n = {len(pick(w, cohort))})", zorder=3)
        for rate, mk in ((0.01, "o"), (0.05, "s")):
            op = ops[rate]
            x = op["fpr"] if op["fpr"] > 0 else zero
            for w, col, ms in (("run1", BLUE, 12), ("sim", ORANGE, 7)):
                ax.plot(x, 100 * op[w], mk, ms=ms, mfc=col, mec="white", mew=1.5, zorder=4)
        ax.axvline(1 / nb, color=INK2, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(zero / 1.3, 1.05)
        ax.set_xticks([zero, 1 / nb, 0.01, 0.05, 0.1, 1], ["0", f"1/{nb}", "1%", "5%", "10%", "100%"])
        ax.minorticks_off()
        ax.set_xlabel("False alarms on honest runs (log scale; lower = better)")
        ax.yaxis.grid(True, color="#e4e3df", zorder=0)
        ax.set_title(f"{name}, agent text removed, full run", loc="left", fontsize=12, color=INK)
        from matplotlib.lines import Line2D

        notes = [Line2D([], [], ls="none", marker=mk, ms=8, mfc=GREY, mec="white", label=f"{rate:.0%} target: {ops[rate]['above']}/{nb} honest flagged")
                 for rate, mk in ((0.01, "o"), (0.05, "s"))]
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + notes, frameon=False, loc="lower right", fontsize=10)
    axes[0].set_ylim(70, 101)  # a line chart, so the axis needn't start at 0; every point below 70% sits at FPR 0
    axes[0].set_ylabel("Attack-instructed runs flagged (%)\nhigher = more runs flagged")
    return fig


def main(stripped: str):
    orig = json.loads((RES / "claim_c_POSTHOC_2026-10-10_225232.json").read_text())["selections"]
    strip = json.loads((RES / stripped).read_text())["selections"]
    OUT.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    from zoneinfo import ZoneInfo

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    for name, fig in [("7_stripped_clean_test", fig6(orig, strip)), ("8_stripped_roc_logfpr", fig7(strip))]:
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_{stamp}.png", dpi=130)
        plt.close(fig)
        print(OUT / f"{name}_{stamp}.png")


if __name__ == "__main__":
    main(sys.argv[1])
