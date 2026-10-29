"""Gap vs per-action false-alarm rate, separate lines, 183 original tasks. Record: e2e_2026-10-15_120037_PA_figure_numbers.json @770323a82 (PAC)."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = sys.argv[1]
SIM, REAL, INK, MUTED = "#eb6834", "#2a78d6", "#222222", "#6b6b68"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED})
P = json.load(open(os.path.join(S, "figs_james", "e2e_2026-10-15_120037_PA_figure_numbers.json")))["PAC"]
PANELS = [("Sol (a)", "Off-policy probe"), ("own_frac100_L48", "On-policy probe (~155 training tasks)")]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, (probe, title) in zip(axes, PANELS):
    ro = "C"
    d = P[ro]["probes"][probe]; x = d["x"]; ok = d["readable"]
    xs = [xi for xi, o in zip(x, ok) if o]
    for key, col, lab in (("yardstick", REAL, "Real run 2 minus real run 1"), ("e2e", SIM, "Simulated minus real")):
        pts = [v for v, o in zip(d[key], ok) if o]
        ax.fill_between(xs, [p[1] for p in pts], [p[2] for p in pts], color=col, alpha=0.18, lw=0)
        ax.plot(xs, [p[0] for p in pts], color=col, lw=2, label=lab)
    floor = d["readable_floor"]
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xscale("log"); ax.set_xlim(floor, max(x))
    ax.set_xticks([3e-3, 1e-2, 3e-2, 1e-1], ["1 in 330", "1 in 100", "1 in 33", "1 in 10"])
    ax.set_xlabel("Honest actions that raise an alarm")
    ax.set_title(title, loc="left", fontsize=12)
axes[0].set_ylabel("Gap in catch rate (percentage points)")
axes[1].legend(frameon=False, loc="lower right", fontsize=10)
fig.text(0.01, 0.01, "Probes scoring commands; 183 original tasks; shaded bands are 95% intervals; each pipeline sets its own alarm line on its own honest runs.\nThe axis stops at the strictest line our honest runs can support (about 1 in 420).", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 1))
out = os.path.join(S, "figs_simple", "fig3_curve.png"); fig.savefig(out, dpi=200); plt.close(fig); print(out)
