"""James's per-task plot: each run's top action score minus its own alarm line, real run 1 vs real run 2 (left) and real run 1 vs simulated (right).
Record: experiments/2026-10-15_fpr_sweep_507_prep/results/pertask455_C_2026-10-15_161408.json @730baf04e (sha256 f374c358...; post hoc; Codex: no mismatches).
Also draws the Spearman summary for every probe (appendix)."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = sys.argv[1]
INK, MUTED, REAL, SIM = "#222222", "#6b6b68", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED})
D = json.load(open(os.path.join(S, "pertask455v2_C.json")))
LAB = {"own_frac100_L48 C": "On-policy, ~155 training tasks", "own_frac25_L48 C": "On-policy, ~39 training tasks", "own_frac10_L48 C": "On-policy, 16 training tasks",
       "Sol (a) C": "Off-policy probe", "FIBS C": "Deception probe", "harmfulness C": "Harmfulness direction", "refusal C": "Refusal direction", "high-stakes C": "High-stakes probe"}


def per_task(probe):
    out = {}
    for origin in ("S_old", "S_new"):
        for e in D["rows"][probe][origin]["per_task"]:
            out.setdefault((origin, e["cell"]), {})[e["arm"]] = e["max_minus_tau"]
    return [v for v in out.values() if {"run1", "run2", "sim"} <= set(v)]


for SHOW, OUT in ((["own_frac100_L48 C"], "fig_pertask_onpolicy.png"), (["own_frac100_L48 C", "Sol (a) C", "harmfulness C"], "fig_pertask.png")):
  plt.rcParams.update({"font.size": 12})
  fig, axes = plt.subplots(len(SHOW), 2, figsize=(10, 4.6 * len(SHOW)), squeeze=False)
  for r, probe in enumerate(SHOW):
      pts = per_task(probe)
      anc, oth = ("run2", "run1") if probe.startswith("own_") else ("run1", "run2")  # on-policy: run 1 is a training arm (2e anchor record @214cfa432)
      x = np.array([p[anc] for p in pts])
      ys = {arm: np.array([p[arm] for p in pts]) for arm in (oth, "sim")}
      lo, hi = np.percentile(np.r_[x, ys[oth], ys["sim"]], [1, 99])
      for c, (arm, col, ylab) in enumerate(((oth, REAL, "Real run " + oth[-1]), ("sim", SIM, "Simulated run"))):
          ax = axes[r, c]; y = ys[arm]
          ax.scatter(x, y, s=10, alpha=0.35, color=col, lw=0)
          b, a = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2
          xx = np.array([lo, hi]); ax.plot(xx, a + b * xx, color=INK, lw=2)
          ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
          ax.axhline(0, color=MUTED, lw=0.9, ls="--", zorder=0); ax.axvline(0, color=MUTED, lw=0.9, ls="--", zorder=0)  # alarm lines: right of / above zero = caught
          ax.text(0.04, 0.95, f"R² = {r2:.2f}", transform=ax.transAxes, va="top", fontsize=13, color=INK, fontweight="bold")
          ax.set_xlabel(f"Real run {anc[-1]}: most suspicious action,\ndistance above the alarm line"); ax.set_ylabel(f"{ylab}: most suspicious action,\ndistance above the alarm line")
          if c == 0:
              ax.set_title(LAB[probe], loc="left", fontsize=13, x=-0.15)
          if r == 0:
              ax.text(0.5, 1.12, ("Two real runs (how well a rerun agrees)" if c == 0 else "Real against simulated"), transform=ax.transAxes, ha="center", fontsize=12, color=INK, fontweight="bold")
  fig.text(0.01, 0.005, "One point per task (all 455 tasks), probes scoring commands. Each axis shows how far a run's most suspicious action sits above (or below)\nits own alarm line; dashed lines mark zero, so points right of or above them were caught. Black line: least-squares fit.\nThe on-policy probe is anchored on real run 2, because real run 1 shares the probe's training launch. Post hoc.", fontsize=9.5, color=MUTED)
  fig.tight_layout(rect=(0, 0.04 if len(SHOW) > 1 else 0.15, 1, 1))
  out = os.path.join(S, "figs_simple", OUT); fig.savefig(out, dpi=170); plt.close(fig); print(out)
plt.rcParams.update({"font.size": 10.5})

# appendix summary: R² for every probe, with intervals (pertask455v2_C_..._r2_2026-10-15_165437.json @867c7e15d)
A = json.load(open(os.path.join(S, "pertask_anchor.json")))["anchor"]
order = ["own_frac100_L48 C", "own_frac25_L48 C", "own_frac10_L48 C", "Sol (a) C", "FIBS C", "harmfulness C", "refusal C", "high-stakes C"]
fig, ax = plt.subplots(figsize=(8, 5))
h = 0.36
for i, probe in enumerate(order):
    for dy, key, col in ((-h / 2, "yardstick", REAL), (h / 2, "main", SIM)):
        v = A[probe]["sets"]["pooled"]["r2"][key]
        ax.barh(i + dy, v["point"], height=h * 0.9, color=col, zorder=2)
        ax.plot(v["ci95"], [i + dy] * 2, color=INK, lw=1, zorder=3)
        ax.text(v["ci95"][1] + 0.01, i + dy, f"{v['point']:.2f}", va="center", fontsize=9)
ax.set_yticks(range(len(order)), [LAB[p] for p in order]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 0.5); ax.set_xlabel("R² with the anchor real run across tasks (higher = more alike)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=REAL, label="The other real run"), Patch(color=SIM, label="Simulated run")], frameon=False, loc="lower right")
fig.text(0.01, 0.01, "All 455 tasks, commands, 1 in 100 honest actions; 95% intervals over resampled tasks. Anchor: real run 2 for on-policy probes (run 1 trains them), real run 1 otherwise. Post hoc.", fontsize=8.8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = os.path.join(S, "figs_simple", "fig_pertask_summary.png"); fig.savefig(out, dpi=200); plt.close(fig); print(out)
