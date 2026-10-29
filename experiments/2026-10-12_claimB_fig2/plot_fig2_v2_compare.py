"""Figure 2, version 2 (labelled comparison): the probe's pooled result beside other readers on the same halves cells
at the same frozen nominal-1% threshold (calibrated on held-out real e1), read by key from their pinned result files.
Pooled only: share of simulated runs and of a second real run (e2, direct) flagged, ±1 binomial SE, and each reader's
gap with its stored 95% task-clustered interval. Rows are added as results land (mask23 when it finishes).

  uv run --no-project --python 3.12 --with matplotlib python experiments/2026-10-12_claimB_fig2/plot_fig2_v2_compare.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
SOURCES = [  # (label, commit, path, how to read pooled sim/e2/gap)
    ("Probe (Qwen3-32B)", "53da7f48", "experiments/2026-10-11_claimB_halves/results/halves_e2_direct_POSTHOC_2026-10-11_172057.json", "probe"),
    ("Llama-3.2-1B-Instruct\n(small LM, same design)†", "28772d47", "experiments/2026-10-12_claimB_1b_halves/results/1b_halves_2026-10-12_130117.json", "llama"),
    ("Text classifier (TF-IDF)", "faafd24c", "experiments/2026-10-12_claimB_tfidf_halves/results/tfidf_halves_POSTHOC_2026-10-12_112423.json", "tfidf"),
    ("Probe, numbers/IDs and whitespace\nmasked in both worlds*", "c3793472", "experiments/2026-10-12_claimB_halves_mask23/results/halves_mask23_2026-10-12_120522.json", "mask23"),
]


def pooled(kind, d):
    if kind == "probe":
        p = d["results"]["pooled"]
        return p["sim_catch"], p["direct"]["e2_flagged_by_sim_detector"], p["direct"]["gap"], p["direct"]["gap_ci95"], 939, 939
    if kind == "llama":
        r = d["results"]["converged_only"]["llama"]
        (cs, ns), (ce, ne) = r["counts"]["pooled"]["sim"], r["counts"]["pooled"]["e2"]
        return cs / ns, ce / ne, r["rates"]["pooled"]["gap"], r["gap_ci95"]["pooled"], ns, ne
    if kind == "mask23":
        p = d["results"]["pooled"]["catch_1pct"]
        return p["sim_catch"], p["direct_e2_flagged"], p["direct_gap"], p["direct_gap_ci95"], 939, 939
    p = d["results"]["pooled"]
    return p["sim"], p["e2"], p["gap"], p["gap_ci95"], p["n_sim"], p["n_e2"]


rows, inputs = [], {}
for label, commit, path, kind in SOURCES:
    raw = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout
    inputs[f"{path}@{commit}"] = hashlib.sha256(raw).hexdigest()
    sim, e2, gap, ci, ns, ne = pooled(kind, json.loads(raw))
    cs, ce = sim * ns, e2 * ne
    assert abs(cs - round(cs)) < 1e-6 and abs(ce - round(ce)) < 1e-6
    rows.append({"label": label, "sim_pct": 100 * sim, "e2_pct": 100 * e2, "sim_flagged": round(cs), "e2_flagged": round(ce), "n": ns,
                 "gap_pts": 100 * gap, "gap_ci95_pts": [100 * ci[0], 100 * ci[1]]})
    print(json.dumps(rows[-1]))

fig, ax = plt.subplots(figsize=(10.5, 5.0))
for i, r in enumerate(rows):
    for j, (pct, cnt, col, lab) in enumerate(((r["sim_pct"], r["sim_flagged"], "#c0392b", "Simulated (fixed sim)"),
                                              (r["e2_pct"], r["e2_flagged"], "#7f8c8d", "Second real run (e2)"))):
        p = pct / 100
        se = 100 * (p * (1 - p) / r["n"]) ** 0.5
        xx = i + (j - 0.5) * 0.36
        ax.bar(xx, pct, 0.34, color=col, label=lab if i == 0 else None)
        ax.errorbar(xx, pct, yerr=se, color="black", capsize=3, lw=1)
        ax.text(xx, pct + se + 0.2, f"{pct:.1f}%\n{cnt}/{r['n']}", ha="center", va="bottom", fontsize=9)
    ax.text(i, max(r["sim_pct"], r["e2_pct"]) + 2.2, f"gap {r['gap_pts']:+.1f} pts\n[{r['gap_ci95_pts'][0]:+.1f}, {r['gap_ci95_pts'][1]:+.1f}]",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(rows)), [r["label"] for r in rows])
ax.set_ylabel("Share flagged (%)")
ax.set_ylim(0, 11)
ax.set_title("Share flagged at the same alarm, same cells (pooled)")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.005, "Environment halves, pooled (939 cells). Threshold: nominal 1%, calibrated on held-out real e1 runs.\n"
         "Bars ±1 binomial SE; gaps with 95% task-clustered intervals. POST HOC; preliminary until reviewed.\n"
         "† Plant-2 gate passed in this run (0.912, 0.970); its earlier level-1 planted checks failed (0.637 vs 0.95; 0.862 vs 0.90).\n"
         "* Descriptive only: its dev→test planted check failed (AUROC 0.893 < 0.910), so its plan gives no reading.\n"
         "Sources: " + "; ".join(f"{p.split('/')[-1]} @ {c}" for _, c, p, _ in SOURCES), fontsize=6.5, color="#555")
fig.tight_layout(rect=(0, 0.15, 1, 1))
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
png = HERE / "results" / f"fig2_v2_compare_{stamp}.png"
fig.savefig(png, dpi=200, bbox_inches="tight")
out = {"label": "Figure 2 v2: probe, Llama-3.2-1B, TF-IDF and probe on masked text (descriptive), pooled, same cells and alarm", "rows": rows, "inputs": inputs, "png": png.name,
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with open(HERE / "results" / f"fig2_v2_compare_{stamp}.json", "x") as fh:
    fh.write(json.dumps(out, indent=1) + "\n")
print("->", png)
