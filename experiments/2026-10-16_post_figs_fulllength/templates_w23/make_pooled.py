"""Pooled 455 tasks, command readout, separate lines. Record pooled455_C_2026-10-15_141921.json @6d2a2fbc5 (keys pooled.<probe>.pooled.joint.<fpr>.{catch,gap}; family.joint.<fpr>)."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
S = sys.argv[1]
SIM, REAL, INK, MUTED = "#eb6834", "#2a78d6", "#222222", "#6b6b68"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED})
d = json.load(open(os.path.join(S, "figs_james", "pooled455_C.json")))
ORDER = [("own_frac100_L48 C", "On-policy, 100% of tasks"), ("own_frac25_L48 C", "On-policy, 25% of tasks"), ("own_frac10_L48 C", "On-policy, 10% of tasks"),
         ("Sol (a) C", "Off-policy probe"), ("FIBS C", "Deception probe"), ("harmfulness C", "Harmfulness direction"),
         ("refusal C", "Refusal direction"), ("high-stakes C", "High-stakes probe")]
# Figure 1: paired catch bars, pooled 455, command, 1e-2
fig, ax = plt.subplots(figsize=(8.5, 6))
h = 0.38; rows = []
for i, (k, lab) in enumerate(ORDER):
    c = d["pooled"][k]["pooled"]["joint"]["0.01"]["catch"]; y = len(ORDER) - 1 - i
    for dy, key, col in ((h / 2, "real_mean", REAL), (-h / 2, "sim", SIM)):
        v, lo, hi = (100 * c[key]["point"], 100 * c[key]["ci95"][0], 100 * c[key]["ci95"][1])
        ax.barh(y + dy, v, h * 0.92, color=col); ax.errorbar(v, y + dy, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, lw=1.1, capsize=2.5)
        ax.text(hi + 1.2, y + dy, f"{v:.0f}", va="center", fontsize=9, color=INK)
    rows.append((lab, 100 * c["real_mean"]["point"], 100 * c["sim"]["point"]))
ax.axhline(len(ORDER) - 3.5, color="#dddddd", lw=1)
ax.set_yticks(range(len(ORDER)), [lab for _, lab in reversed(ORDER)]); ax.set_xlim(0, 110); ax.set_xlabel("Attack runs caught (%)")
ax.legend(handles=[Patch(color=REAL, label="Real attacks (mean of two runs)"), Patch(color=SIM, label="Simulated attacks")], frameon=False, loc="lower right", fontsize=10)
fig.text(0.01, 0.01, "455 tasks; command readout; 95% intervals; each pipeline sets its own alarm line at 1 in 100 honest actions.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(S, "figs_simple", "fig1_pairs455.png"), dpi=200); plt.close(fig)
# Figure 2: family average gap, sim vs rerun, with intervals, at 1e-2 and 5e-2
fam = d["family"]["joint"]
fig, ax = plt.subplots(figsize=(7.5, 4.8)); w = 0.36; vals = {}
for i, (fpr, lab) in enumerate((("0.01", "1 in 100 honest actions"), ("0.05", "1 in 20 honest actions"))):
    for dx, key, col in ((-w / 2, "mean_abs_delta_full", SIM), (w / 2, "mean_abs_delta_rr", REAL)):
        q = fam[fpr][key]; v, lo, hi = 100 * q["point"], 100 * q["ci95"][0], 100 * q["ci95"][1]
        ax.bar(i + dx, v, w * 0.92, color=col); ax.errorbar(i + dx, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, lw=1.2, capsize=4)
        ax.text(i + dx, hi + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
    q = fam[fpr]["difference"]; vals[fpr] = (100 * q["point"], 100 * q["ci95"][0], 100 * q["ci95"][1])
ax.set_xticks([0, 1], ["1 in 100 honest actions", "1 in 20 honest actions"]); 
ax.set_ylabel("Average gap in catch rate\n(percentage points; lower = closer)"); ax.set_ylim(0, 7.5)
ax.legend(handles=[Patch(color=SIM, label="Simulated vs real"), Patch(color=REAL, label="Real run 2 vs real run 1")], frameon=False, loc="upper right")
fig.text(0.01, 0.01, f"455 tasks; command readout; averaged over the eight probes; 95% intervals. Simulated minus rerun: {vals['0.01'][0]:+.1f} [{vals['0.01'][1]:+.1f}, {vals['0.01'][2]:+.1f}] at 1 in 100, {vals['0.05'][0]:+.1f} [{vals['0.05'][1]:+.1f}, {vals['0.05'][2]:+.1f}] at 1 in 20.", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(S, "figs_simple", "fig2_family455.png"), dpi=200); plt.close(fig)
print("pairs:", [(l, round(r, 1), round(s, 1)) for l, r, s in rows]); print("family diff:", {k: tuple(round(x, 2) for x in v) for k, v in vals.items()})
