"""James's MVP figure: real vs simulated catch per probe, 95% intervals. Records: peraction311_all_{C,R} @871a1269d/4f9b57700, peraction311_all_O @0143d6520."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = sys.argv[1]
SIM, REAL, INK, MUTED = "#eb6834", "#2a78d6", "#222222", "#6b6b68"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED})
J = lambda f: json.load(open(os.path.join(S, "figs_james", f)))["rows"]
recs = {ro: json.load(open(os.path.join(S, "figs_q2final", f"v2_{ro}.json")))["rows"] for ro in "CR"}  # Q2-FINAL pooled455v2 @672b1dd1b, S_new rows
own = J("peraction311_all_O.json")
ORDER = [("own_frac100_L48", "On-policy, ~155 training tasks"), ("own_frac25_L48", "On-policy, ~39 training tasks"), ("own_frac10_L48", "On-policy, 16 training tasks"),
         ("Sol (a)", "Off-policy probe (GPT-5.6-Sol's runs)"), ("FIBS", "Deception probe (FAR.AI)"), ("harmfulness", "Harmfulness direction"),
         ("refusal", "Refusal direction"), ("high-stakes", "High-stakes probe")]
def get(key, ro):
    c = recs[ro][f"{key} {ro}"]["S_new"]["points"]["joint"]["0.01"]["catch"]
    f = lambda k: (100 * c[k]["point"], 100 * c[k]["ci95"][0], 100 * c[k]["ci95"][1])
    return f("real_mean@B"), f("sim@S|primary")
fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
h = 0.38
for ax, ro, title in ((axes[0], "C", "Command readout"), (axes[1], "R", "Tool-output readout")):
    for i, (key, lab) in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        (r, rlo, rhi), (s, slo, shi) = get(key, ro)
        for dy, (v, lo, hi), col in ((h / 2, (r, rlo, rhi), REAL), (-h / 2, (s, slo, shi), SIM)):
            ax.barh(y + dy, v, h * 0.92, color=col)
            ax.errorbar(v, y + dy, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, lw=1.1, capsize=2.5)
            ax.text(hi + 1.2, y + dy, f"{v:.0f}", va="center", fontsize=9, color=INK)
    ax.axhline(len(ORDER) - 3.5, color="#dddddd", lw=1)
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlim(0, 112); ax.set_xlabel("Attack runs caught (%)")
axes[0].set_yticks(range(len(ORDER)), [lab for _, lab in reversed(ORDER)])
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=REAL, label="Real attacks (mean of two runs)"), Patch(color=SIM, label="Simulated attacks")], frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.99))
fig.suptitle("Catch rate on real and simulated attacks", x=0.01, ha="left", fontsize=13)
fig.text(0.01, 0.01, "272 new tasks; 95% intervals; each pipeline sets its own alarm line at 1 in 100 honest actions. Top three: probes trained on Qwen3.5's own runs.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.94)); fig.savefig(os.path.join(S, "figs_simple", "fig1_pairs.png"), dpi=200); plt.close(fig); print("ok")
