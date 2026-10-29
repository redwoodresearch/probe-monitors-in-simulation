"""Figures for PLAN.md's two post hoc checks, from their result files (numbers only).

  uv run --no-project --python 3.13 --with matplotlib python experiments/2026-10-11_claimB_q11_posthoc/plot_POSTHOC_q11.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def save(fig, path: Path):
    """results/ is append-only: refuse to overwrite an existing figure (rename or remove it deliberately first)."""
    if path.exists():
        raise SystemExit(f"{path} exists; results/ is append-only")
    fig.savefig(path, dpi=150)

HERE = Path(__file__).parent
NULL = HERE / "results/stepA_catch_null_POSTHOC_2026-10-11_040431.json"
GAP = HERE / "results/probe_vs_tfidf_POSTHOC_2026-10-11_040340.json"
NAMES = {"fixed_vs_e1": "fixed sim vs real\n(step A's reading)", "e2_vs_e1": "real vs real\n(control)",
         "plant1": "planted\ncheck 1", "plant2": "planted\ncheck 2"}

rows = json.loads(NULL.read_text())["rows"]
fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(rows))
obs = [100 * r["at_1pct"]["observed"] for r in rows]
p95 = [100 * r["at_1pct"]["null_p95"] for r in rows]
ax.bar([i - 0.2 for i in x], obs, 0.4, label="caught (real labels)", color="#2b6cb0")
ax.bar([i + 0.2 for i in x], p95, 0.4, label="shuffle null, 95th percentile", color="#a0aec0")
for i, r in enumerate(rows):
    a = r["at_1pct"]
    ax.text(i - 0.2, obs[i] + 0.8, f"{obs[i]:.1f}%\np={a['p']:.3f}", ha="center", fontsize=9)
    ax.text(i + 0.2, p95[i] + 0.8, f"{p95[i]:.1f}%", ha="center", fontsize=9)
ax.set_xticks(list(x), [NAMES[r["comparison"]] for r in rows])
ax.set_ylabel("share caught at a 1% false-alarm target (%)\nhigher = more caught")
ax.set_ylim(0, max(obs) * 1.25)
ax.set_title("Step A's probe: catch at 1% against its own shuffle null (level 1, 5 test environments)", fontsize=10)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
save(fig, HERE / "results/stepA_catch_null_POSTHOC_2026-10-11_040431.png")

g = json.loads(GAP.read_text())["rows"]
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for i, (k, label) in enumerate((("fixed_vs_e1", "fixed sim vs real"), ("e2_vs_e1", "real vs real (control)"))):
    r = g[k]
    ax.errorbar(i, r["difference"], yerr=[[r["difference"] - r["ci95"][0]], [r["ci95"][1] - r["difference"]]], fmt="o",
                color="#2b6cb0" if i == 0 else "#718096", capsize=6, ms=8)
    ax.text(i + 0.08, r["difference"], f"{r['difference']:+.3f}\n[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]", va="center", fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks([0, 1], ["fixed sim vs real", "real vs real\n(control)"])
ax.set_xlim(-0.5, 1.9)
ax.set_ylabel("AUROC(probe) − AUROC(TF-IDF)\nhigher = probe separates better")
ax.set_title("Probe vs TF-IDF on the same 946 runs (95% task bootstrap)", fontsize=10)
fig.tight_layout()
save(fig, HERE / "results/probe_vs_tfidf_POSTHOC_2026-10-11_040340.png")
print("ok")
