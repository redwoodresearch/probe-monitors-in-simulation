"""James's MVP figure: real vs simulated catch per probe, 95% intervals. Records: peraction311_all_{C,R} @871a1269d/4f9b57700, peraction311_all_O @0143d6520."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = sys.argv[1]
SIM, REAL, INK, MUTED = "#eb6834", "#2a78d6", "#222222", "#6b6b68"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED})
J = lambda f: json.load(open(os.path.join(S, "figs_james", f)))["rows"]
ALL = json.load(open(os.path.join(S, "alltasks_CRD.json")))["pooled"]  # alltasks_CRD_2026-10-15_161306 @6acf23b06, pooled_all, joint intervals
own = J("peraction311_all_O.json")
ORDER = [("own_frac100_L48", "On-policy probe (Qwen3.5's own runs)"), 
         ("Sol (a)", "Off-policy probe (GPT-5.6-Sol's runs)"), ("FIBS", "Deception probe (FAR.AI)"), ("harmfulness", "Harmfulness direction"),
         ("refusal", "Refusal direction"), ("high-stakes", "High-stakes probe")]
def get(key, ro):
    j = ALL[f"{key} {ro}"]["pooled_all"]["joint"]["0.01"]
    c = j["catch"]
    f = lambda k: (100 * c[k]["point"], 100 * c[k]["ci95"][0], 100 * c[k]["ci95"][1])
    return f("real_mean"), f("sim")
fig, ax0 = plt.subplots(1, 1, figsize=(8.5, 4.9)); axes = [ax0]
h = 0.38
for ax, ro, title in ((axes[0], "C", "Command readout"),):
    for i, (key, lab) in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        (r, rlo, rhi), (s, slo, shi) = get(key, ro)
        for dy, (v, lo, hi), col in ((h / 2, (r, rlo, rhi), REAL), (-h / 2, (s, slo, shi), SIM)):
            ax.barh(y + dy, v, h * 0.92, color=col)
            ax.errorbar(v, y + dy, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, lw=1.1, capsize=2.5)
            ax.text(hi + 1.2, y + dy, f"{v:.0f}", va="center", fontsize=9, color=INK)
    ax.axhline(len(ORDER) - 1.5, color="#dddddd", lw=1)
    pass
    ax.set_xlim(0, 112); ax.set_xlabel("Attack runs caught (%)")
NS = {k: int(sum(ALL[f"{k} C"]["pooled_all"]["joint"]["0.01"]["D_draw0"].values())) for k, _ in ORDER}
axes[0].set_yticks(range(len(ORDER)), [f"{lab}\n{NS[k]} tasks" for k, lab in reversed(ORDER)])
from matplotlib.patches import Patch
ax0.legend(handles=[Patch(color=REAL, label="Real attacks (mean of two runs)"), Patch(color=SIM, label="Simulated environment")], frameon=False, loc="lower right", fontsize=9.5)
fig.suptitle("Catch rate on commands, real and simulated", x=0.01, ha="left", fontsize=13)
fig.text(0.01, 0.01, "All tasks each probe covers; 95% intervals over resampled tasks, alarm lines redrawn each time;\neach pipeline sets its own alarm line at 1 in 100 honest actions. Post hoc.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 0.94)); fig.savefig(os.path.join(S, "figs_simple", "fig1_pairs_all.png"), dpi=200); plt.close(fig); print("ok")
