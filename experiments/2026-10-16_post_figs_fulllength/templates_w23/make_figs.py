"""Simple figures for the probe-shift post. Reads committed records (copied locally); light plotting only.
Sources: p311 figure numbers @86d5fea80; peraction311_all_O @0143d6520; il_final @d851974f5."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = sys.argv[1]  # scratchpad dir
OUT = os.path.join(S, "figs_simple")
SIM, RR, INK, MUTED = "#eb6834", "#2a78d6", "#222222", "#6b6b68"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})

p311 = json.load(open(os.path.join(S, "figs_james/p311_2026-10-15_133728_P311_figure_numbers.json")))["1e-2"]["rows"]
own = json.load(open(os.path.join(S, "figs_james/peraction311_all_O.json")))["rows"]
il = json.load(open(os.path.join(S, "figs_simple/il_final.json")))["result_il"]["append36"]["quantities_pp"]

NAMES = {"Sol (a)": "Off-policy probe (GPT-5.6-Sol's runs)", "FIBS": "Deception probe (FAR.AI)", "harmfulness": "Harmfulness direction",
         "refusal": "Refusal direction", "high-stakes": "High-stakes probe",
         "own_frac100_L48": "Own runs, 100% of tasks", "own_frac25_L48": "Own runs, 25% of tasks", "own_frac10_L48": "Own runs, 10% of tasks"}
OFF = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes"]
ON = ["own_frac100_L48", "own_frac25_L48", "own_frac10_L48"]

def off_row(name, ro):
    r = next(x for x in p311 if x["row"] == f"{name} {ro}")
    return r["delta_full_joint"], r["delta_rr_joint"]  # [point, lo, hi] in points

def on_row(name, ro):
    g = own[f"{name} {ro}"]["points"]["joint"]["0.01"]["gap"]
    f = lambda k: [100 * g[k]["point"], 100 * g[k]["ci95"][0], 100 * g[k]["ci95"][1]]
    return f("delta_full"), f("delta_rr")

def mean_abs(rows):
    s = [abs(a[0]) for a, b in rows]; r = [abs(b[0]) for a, b in rows]
    return sum(s) / len(s), sum(r) / len(r)

# ---------- Figure 1: average gap, simulation vs a second real run ----------
groups = [("On-policy,\ncommand", [on_row(n, "C") for n in ON]), ("On-policy,\ntool output", [on_row(n, "R") for n in ON]),
          ("Off-policy and\npublished, command", [off_row(n, "C") for n in OFF]), ("Off-policy and\npublished, tool output", [off_row(n, "R") for n in OFF])]
vals = [mean_abs(rows) for _, rows in groups]
pts = [([abs(x[0][0]) for x in rows], [abs(x[1][0]) for x in rows]) for _, rows in groups]
fig, ax = plt.subplots(figsize=(9, 5))
w = 0.36
for i, (s, r) in enumerate(vals):
    for dx, v, c in ((-w / 2, s, SIM), (w / 2, r, RR)):
        ax.bar(i + dx, v, w * 0.92, color=c)
        ax.text(i + dx, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=13, color=INK, fontweight="bold", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=11)
ax.set_ylabel("Average gap in catch rate\n(percentage points; lower = closer)")
ax.bar(0, 0, color=SIM, label="Simulated vs real"); ax.bar(0, 0, color=RR, label="Real run 2 vs real run 1")
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(0, max(max(v) for v in vals) * 1.3)
fig.text(0.01, 0.005, "Each bar averages the absolute gap over the probes in its group. 272 new tasks; each evaluation sets its own alarm line at 1 in 100 honest actions. \nOn-policy = trained on Qwen3.5's own runs (3 versions). Off-policy and published = the probe trained on GPT-5.6-Sol's runs plus 4 published probes.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(OUT, "fig1_average_gap.png"), dpi=200); plt.close(fig)

# ---------- Figure 2: per-probe gaps with 95% intervals ----------
order = ON + OFF
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
n_excl = 0
for ax, ro, title in ((axes[0], "C", "Command readout"), (axes[1], "R", "Tool-output readout")):
    for i, n in enumerate(order):
        a, b = (on_row if n in ON else off_row)(n, ro)
        y = len(order) - 1 - i
        n_excl += (a[1] > 0 or a[2] < 0)
        ax.errorbar(a[0], y + 0.15, xerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="o", color=SIM, ms=7, lw=2, capsize=0)
        ax.errorbar(b[0], y - 0.15, xerr=[[b[0] - b[1]], [b[2] - b[0]]], fmt="o", color=RR, ms=7, lw=2, capsize=0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axhline(len(OFF) - 0.5, color="#dddddd", lw=1)
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("Gap in catch rate (percentage points)")
    ax.set_xlim(-22, 24)
axes[0].set_yticks(range(len(order)), [NAMES[n] for n in reversed(order)])
axes[1].errorbar([], [], fmt="o", color=SIM, label="Simulated minus real"); axes[1].errorbar([], [], fmt="o", color=RR, label="Real run 2 minus real run 1")
fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2, fontsize=10, bbox_to_anchor=(0.99, 0.94))
fig.text(0.01, 0.01, "272 new tasks; 95% intervals; each evaluation sets its own alarm line at 1 in 100 honest actions. "
         "Top three rows: probes trained on Qwen3.5's own runs.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.90)); fig.savefig(os.path.join(OUT, "fig2_per_probe.png"), dpi=200); plt.close(fig)

# ---------- Figure 3: launching together ----------
bars = [("Simulated runs from\na later launch", il["Delta_sep"]), ("Simulated and real runs\nfrom the same launch", il["Delta_same"])]
fig, ax = plt.subplots(figsize=(6.5, 5))
for i, (lab, q) in enumerate(bars):
    v, lo, hi = q["point"], q["ci95"][0], q["ci95"][1]
    ax.bar(i, v, 0.55, color=SIM)
    ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, lw=1.5, capsize=5)
    ax.text(i + 0.30, v, f"{v:+.1f}\n[{lo:+.1f}, {hi:+.1f}]", va="center", ha="left", fontsize=11, color=INK)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(range(2), [b for b, _ in bars]); ax.set_xlim(-0.6, 1.9)
ax.set_ylabel("Simulated minus real catch\n(percentage points)")
fig.text(0.01, 0.01, "Probe trained on GPT-5.6-Sol's runs; 257 new tasks; tool-output readout; 95% intervals.\nEarlier metric: one line set on real honest runs, 5% of honest runs flagged.", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(OUT, "fig3_launch.png"), dpi=200); plt.close(fig)

print("fig1 values (sim, rerun):", [(round(s, 2), round(r, 2)) for s, r in vals]); print("fig2 intervals excluding 0:", n_excl)
