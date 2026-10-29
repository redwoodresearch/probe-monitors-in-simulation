"""Simulated minus real catch against alarm strictness, newer tasks (final honest runs), probes scoring commands.
Record: 2026-10-15_sweep_curve_figure_numbers.json @ebfeb8064 (points keyed to sweep_final_C_2026-10-15_160740.json, sha e828f365; joint intervals)."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = sys.argv[1]
INK, MUTED, SIM, RR = "#222222", "#6b6b68", "#eb6834", "#8d8d88"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED})
P = json.load(open(os.path.join(S, "sweep_final_numbers.json")))["points"]
# rerun band: 2e export @d87ce5e7c (sha256 133f86a7...), rows keyed gap.delta_rr, same records as the curve
Y = [r for r in json.load(open(os.path.join(S, "yardstick_S_new.json")))["rows"] if r["readout"] == "command"]
PANELS = [("own_frac100_L48 C", "On-policy probe"), ("Sol (a) C", "Off-policy probe"), ("FIBS C", "Deception probe"),
          ("harmfulness C", "Harmfulness direction"), ("refusal C", "Refusal direction"), ("high-stakes C", "High-stakes probe")]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharey=True); axes = axes.ravel()
for ax, (lab, title) in zip(axes, PANELS):
    pts = sorted((float(p["level"]), p) for p in P if p["readout"] == "command" and p["origin"] == "S_new" and p["label"] == lab)
    x = [l for l, _ in pts]; y = [100 * p["point"] for _, p in pts]
    lo = [100 * p["ci95"][0] for _, p in pts]; hi = [100 * p["ci95"][1] for _, p in pts]
    rr = sorted((float(r["level"]), r) for r in Y if r["label"] == lab)
    rx = [l for l, _ in rr]
    ax.fill_between(rx, [100 * r["ci95"][0] for _, r in rr], [100 * r["ci95"][1] for _, r in rr], color=RR, alpha=0.22, lw=0)
    ax.plot(rx, [100 * r["point"] for _, r in rr], color=RR, lw=1.5, ls="--")
    ax.fill_between(x, lo, hi, color=SIM, alpha=0.18, lw=0)
    ax.plot(x, y, color=SIM, lw=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xscale("log"); ax.set_xlim(min(x), max(x))
    ax.set_xticks([1e-3, 1e-2, 5e-2], ["1 in 1,000", "1 in 100", "1 in 20"])
    ax.minorticks_off()
    ax.set_title(title, loc="left", fontsize=14)
    if lab == "own_frac100_L48 C":  # values read by key: sweep +10.7, yardstick delta_rr +16.5 at 1e-3
        ax.annotate("+10.7", (1e-3, 10.7), xytext=(1.45e-3, 21), color=SIM, fontsize=11, arrowprops=dict(arrowstyle="-", color=SIM, lw=0.8))
        ax.annotate("+16.5", (1e-3, 16.5), xytext=(1.45e-3, 26), color=MUTED, fontsize=11, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("Honest actions that raise an alarm")
axes[0].set_ylabel("Catch-rate gap (points)"); axes[3].set_ylabel("Catch-rate gap (points)")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=SIM, lw=2, label="Simulated minus real run"), Line2D([], [], color=RR, lw=1.5, ls="--", label="Second real run minus first (rerun)")], frameon=False, loc="upper right", ncol=2, fontsize=12)
fig.text(0.01, 0.01, "272 tasks with enough simulated honest runs, probes scoring commands, each pipeline on its own alarm line.\nShaded bands are 95% intervals over resampled tasks, with alarm lines redrawn each time. Post hoc.", fontsize=11, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=3.0)
out = os.path.join(S, "figs_simple", "fig_sweep_final.png"); fig.savefig(out, dpi=200); plt.close(fig); print(out)
