"""Per-task RMS bias (the simulated control evaluations paper's method), commands, 1 alarm per 100 honest actions, separate lines,
with a real-versus-real reference (real run 3 against real runs 1-2), per Jeff via 09.
Record: experiments/2026-10-15_il_analysis_09/results/il_posthoc_peraction_2026-10-15_151554.json @2fa8631bb (09-verified, post hoc, preliminary).
Keys: rms_bias_per_action.<set>.rows["separate|0.01|C"].per_probe_all_readable[<probe>].
  {task_pair_level_scenario_A, real_vs_real_reference_run3}.{rms_bias_signed_unfloored, rms_bias_signed_ci95}."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = sys.argv[1]
INK, MUTED, SIMC, REFC = "#222222", "#6b6b68", "#eb6834", "#9a9a96"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": MUTED})
R = json.load(open(os.path.join(S, "rms_peraction.json")))["rms_bias_per_action"]
SETS = [("S_old", "Original tasks (88)\nreal runs from different launches"), ("S_new", "Newer tasks (257)"), ("S_newmains", "Newest tasks (239)")]
PROBES = [("Sol (a) C", "Off-policy probe"), ("FIBS C", "Deception probe"), ("harmfulness C", "Harmfulness direction"),
          ("refusal C", "Refusal direction"), ("high-stakes C", "High-stakes probe")]
fig, axes = plt.subplots(1, 3, figsize=(13, 5.4), sharey=True)
h = 0.36
for ax, (st, title) in zip(axes, SETS):
    rows = R[st]["rows"]["separate|0.01|C"]["per_probe_all_readable"]
    for i, (key, lab) in enumerate(PROBES):
        v = rows.get(key)
        if v is None:
            ax.text(-38, i, "pending" if st == "S_new" else "not covered", fontsize=9, color=MUTED, va="center")
            continue
        for dy, blk, col in ((-h / 2, "task_pair_level_scenario_A", SIMC), (h / 2, "real_vs_real_reference_run3", REFC)):
            b = v[blk]; p = 100 * b["rms_bias_signed_unfloored"]; lo, hi = (100 * x for x in b["rms_bias_signed_ci95"])
            ax.barh(i + dy, p, height=h * 0.9, color=col, zorder=2)
            ax.plot([lo, hi], [i + dy, i + dy], color=INK, lw=1.0, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-42, 42); ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(PROBES)), [lab for _, lab in PROBES]); axes[0].invert_yaxis()
axes[1].set_xlabel("RMS bias on commands (points)", fontsize=11)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=SIMC, label="Simulated against real runs"), Patch(color=REFC, label="Real run 3 against real runs 1 and 2 (reference)")],
           frameon=False, loc="upper right", ncol=2, fontsize=10, bbox_to_anchor=(0.99, 1.0))
fig.text(0.01, 0.01, "Bias = square root of (mean squared difference minus each world's run-to-run variance), signed by that difference; negative means less disagreement than noise predicts.\n"
         "1 in 100 honest actions, separate lines; 95% intervals over tasks; task-to-task spread 11 to 26 points. High-stakes on the original tasks gives a negative estimate, which sampling noise alone can produce. Post hoc.", fontsize=8.6, color=MUTED)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
out = os.path.join(S, "figs_simple", "fig_rms.png"); fig.savefig(out, dpi=200); plt.close(fig); print(out)
