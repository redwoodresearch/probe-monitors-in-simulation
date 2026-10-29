"""Simulation gap minus rerun gap, all 455 tasks, per probe, one panel per readout, at 1 alarm per 100 honest actions.
Combination: each task set's gaps are computed on their own, then averaged weighted by task count (pooled_abs_per_origin),
so simulation errors of opposite sign in the two sets cannot cancel.
Record: experiments/2026-10-15_panel_noise_matched/results/panel_pooled455_RC_2026-10-15_145109.json @93e0ad177 (FINAL)."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = sys.argv[1]
INK, MUTED, BAR, SUM = "#222222", "#6b6b68", "#8a8fc4", "#3b3f7a"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": MUTED})
Z = json.load(open(os.path.join(S, "panel455_v6.json")))["results"]["pooled"]["0.01"]  # panel v6 @9238c1ff9, pooled first (Aswin 19:55); within-batch version is make_excess_C_within.py
PROBES = [("own_frac100_L48", "On-policy probe"), 
          ("Sol (a)", "Off-policy probe"), ("FIBS", "Deception probe"), ("harmfulness", "Harmfulness direction"),
          ("refusal", "Refusal direction"), ("high-stakes", "High-stakes probe")]

fig, ax0 = plt.subplots(1, 1, figsize=(8, 5.6)); axes = [ax0]
for ax, (ro, title) in zip(axes, (("C", "Command readout"),)):
    rows = [("Average of these six", Z["groups"][f"body6|{ro}"]["primary"]["mean_G_minus_mean_Y"], True)]
    rows += [(lab, Z["rows"][f"{k} {ro}"]["G_minus_Y"], False) for k, lab in PROBES]
    ys = []
    for i, (lab, v, summary) in enumerate(rows):
        y = i + (0 if summary else 0.6)
        p, (lo, hi) = v["point"], v["ci95"]
        ax.barh(y, p, height=0.62, color=SUM if summary else BAR, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.3, zorder=3)
        ax.text(hi + 0.6, y, f"{p:+.1f}", fontsize=10.5, va="center", ha="left", color=INK, fontweight="bold" if summary else "normal")
        ys.append(y)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.set_xlim(-13, 15)
    ax.set_title("Probes scoring commands", loc="left", fontsize=13)
    ax.set_xlabel("Simulation gap minus rerun gap (points)\n← simulation closer to real · farther →", fontsize=10.5)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(ys, [r[0] for r in rows])
axes[0].invert_yaxis()
axes[0].get_yticklabels()[0].set_fontweight("bold")
fig.text(0.01, 0.01, "All 455 tasks pooled; 1 alarm per 100 honest actions, each pipeline on its own line.\nLines: 95% intervals over resampled tasks. Measured within each batch instead, see the appendix.", fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0.1, 1, 1))
out = os.path.join(S, "figs_simple", "fig2_excess_C.png"); fig.savefig(out, dpi=200); plt.close(fig); print(out)
