"""Simulation gap minus rerun gap, per probe and readout, each task set separately, at 1 alarm per 100 honest actions.
Record: experiments/2026-10-15_panel_noise_matched/results/panel_pooled455_RC_2026-10-15_145109.json @93e0ad177 (FINAL; results identical to @7eb9462c1; G = mean |sim - each real run|, Y = |real 2 - real 1|; 95% percentile intervals over task resamples)."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = sys.argv[1]
INK, MUTED, BAR, SUM = "#222222", "#6b6b68", "#7a7fb8", "#3b3f7a"
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": MUTED})
R = json.load(open(os.path.join(S, "panel455_v4.json")))["results"]
PROBES = [("own_frac100_L48", "On-policy, ~155 training tasks"), ("own_frac25_L48", "On-policy, ~39 training tasks"), ("own_frac10_L48", "On-policy, 16 training tasks"),
          ("Sol (a)", "Off-policy probe"), ("FIBS", "Deception probe"), ("harmfulness", "Harmfulness direction"),
          ("refusal", "Refusal direction"), ("high-stakes", "High-stakes probe")]
SETS = [("S_old", "183 original tasks"), ("S_new", "272 new tasks")]


def rows_for(s):
    z = R[s]["0.01"]
    g = lambda key: z["groups"][key]["primary"]["mean_G_minus_mean_Y"]
    out = [("All probes, both readouts", g("overall|R+C"), True, None)]
    for ro, head in (("C", "Command readout"), ("R", "Tool-output readout")):
        out.append((f"All probes", g(f"overall|{ro}"), True, head))
        out += [(lab, z["rows"][f"{k} {ro}"]["G_minus_Y"], False, None) for k, lab in PROBES]
    return out


fig, axes = plt.subplots(1, 2, figsize=(13, 8.2), sharey=True)
for ax, (s, title) in zip(axes, SETS):
    rows = rows_for(s)
    y, ys, labels = 0, [], []
    for lab, v, summary, head in rows:
        if head:
            y += 0.6
            ax.text(-25.5, y, head, fontsize=10.5, fontweight="bold", color=INK, va="center", ha="left", clip_on=False) if s == "S_old" else None
            y += 1
        p, (lo, hi) = v["point"], v["ci95"]
        col = SUM if summary else BAR
        ax.barh(y, p, height=0.62, color=col, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.2, zorder=3)
        ax.text(1.02, y, f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]", transform=ax.get_yaxis_transform(), clip_on=False, fontsize=9, va="center", ha="left", color=INK, fontweight="bold" if summary else "normal")
        ys.append(y); labels.append(lab)
        y += 1
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(5, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_xlim(-18, 17)
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("Simulation gap minus rerun gap (points)\n← simulation closer to real than a rerun · simulation farther →", fontsize=9.5)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(ys, labels)
axes[0].invert_yaxis()
for t in axes[0].get_yticklabels():
    if t.get_text().startswith("All probes"):
        t.set_fontweight("bold")
fig.text(0.01, 0.01, "1 alarm per 100 honest actions; each pipeline sets its own line. Bars show the point estimate, lines the 95% interval over resampled tasks. Dashed line: 5 points.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.subplots_adjust(wspace=0.42, right=0.88)
out = os.path.join(S, "figs_simple", "fig2_excess.png"); fig.savefig(out, dpi=200); plt.close(fig); print(out)
