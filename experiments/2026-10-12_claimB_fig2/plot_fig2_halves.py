"""Figure 2 regenerated from raw data (Aswin's write-up list, "Final figures regenerated from raw data"; via 0d, 10-12).
Reads the halves direct-yardstick result by key from its pinned commit, prints every value, checks them against the
numbers in the outline's image 0b7a4a3b, and renders the bar chart: share of simulated runs and of a second real run
(e2) flagged by the sim-trained probe at the frozen nominal-1% threshold calibrated on held-out real e1 runs (it flags
about 0.85% of e1 by construction; e2's 1.5–1.7% is the independent real-run rate); dev → test, test →
dev and pooled. Bars: share flagged with ±1 binomial SE; above each pair, the gap (sim − e2) and its stored 95%
task-clustered interval. No computation beyond the counts and the SE.

  uv run --no-project --python 3.12 --with matplotlib python experiments/2026-10-12_claimB_fig2/plot_fig2_halves.py
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
PIN, SRC = "53da7f48", "experiments/2026-10-11_claimB_halves/results/halves_e2_direct_POSTHOC_2026-10-11_172057.json"
N = {"dev_to_test": 473, "test_to_dev": 466, "pooled": 939}  # scored test cells per direction (halves PLAN: 473 test, 466 dev)
IMAGE = {"dev_to_test": (7.0, 1.7, 5.3, 2.3, 9.7), "test_to_dev": (3.9, 1.5, 2.4, 0.0, 7.8), "pooled": (5.4, 1.6, 3.8, 1.9, 7.0)}
NAMES = {"dev_to_test": "Train dev, test test", "test_to_dev": "Train test, test dev", "pooled": "Pooled"}

raw = subprocess.run(["git", "show", f"{PIN}:{SRC}"], capture_output=True, check=True, cwd=HERE).stdout
d = json.loads(raw)["results"]
rows, mismatches = {}, []
for k in N:
    sim, e2 = d[k]["sim_catch"], d[k]["direct"]["e2_flagged_by_sim_detector"]
    gap, (lo, hi) = d[k]["direct"]["gap"], d[k]["direct"]["gap_ci95"]
    cs, ce = sim * N[k], e2 * N[k]
    assert abs(cs - round(cs)) < 1e-6 and abs(ce - round(ce)) < 1e-6, f"{k}: rates are not counts over {N[k]}"
    got = tuple(round(100 * x, 1) for x in (sim, e2, gap, lo, hi))
    rows[k] = {"sim_flagged": int(round(cs)), "e2_flagged": int(round(ce)), "n": N[k], "sim_pct": 100 * sim, "e2_pct": 100 * e2,
               "gap_pts": 100 * gap, "gap_ci95_pts": [100 * lo, 100 * hi], "rounded": got, "image": IMAGE[k], "matches_image": got == IMAGE[k]}
    if got != IMAGE[k]:
        mismatches.append(k)
    print(k, json.dumps(rows[k]))

fig, ax = plt.subplots(figsize=(7.5, 4.6))
x = range(len(N))
for i, k in enumerate(N):
    r = rows[k]
    for j, (pct, cnt, col, lab) in enumerate(((r["sim_pct"], r["sim_flagged"], "#c0392b", "Simulated (fixed sim)"),
                                              (r["e2_pct"], r["e2_flagged"], "#7f8c8d", "Second real run (e2)"))):
        p = pct / 100
        se = 100 * (p * (1 - p) / r["n"]) ** 0.5
        xx = i + (j - 0.5) * 0.36
        ax.bar(xx, pct, 0.34, color=col, label=lab if i == 0 else None)
        ax.errorbar(xx, pct, yerr=se, color="black", capsize=3, lw=1)
        ax.text(xx, pct + se + 0.25, f"{pct:.1f}%\n{cnt}/{r['n']}", ha="center", va="bottom", fontsize=9)
    ax.text(i, max(r["sim_pct"], r["e2_pct"]) + 2.6, f"gap {r['gap_pts']:+.1f} pts\n[{r['gap_ci95_pts'][0]:+.1f}, {r['gap_ci95_pts'][1]:+.1f}]",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(list(x), [NAMES[k] for k in N])
ax.set_ylabel("Share flagged (%)")
ax.set_ylim(0, 14)
ax.set_title(f"Probe flags {rows['pooled']['sim_pct']:.1f}% of simulated runs vs {rows['pooled']['e2_pct']:.1f}% of a second real run")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "Qwen3-32B probe, environment halves. Higher red than grey = the probe tells simulated from real.\n"
         "Threshold: nominal 1%, calibrated on held-out real e1 runs. "
         f"Bars ±1 binomial SE; gaps with 95% task-clustered intervals.\nSource: {SRC.split('/')[-1]} @ {PIN}",
         fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.08, 1, 1))
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
png = HERE / "results" / f"fig2_halves_{stamp}.png"
fig.savefig(png, dpi=200, bbox_inches="tight")
out = {"label": "Figure 2 regenerated from raw data", "source": f"{SRC}@{PIN}", "source_sha256": hashlib.sha256(raw).hexdigest(),
       "rows": rows, "mismatches_vs_image_0b7a4a3b": mismatches, "png": png.name,
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with open(HERE / "results" / f"fig2_halves_{stamp}.json", "x") as fh:
    fh.write(json.dumps(out, indent=1) + "\n")
print("mismatches vs image:", mismatches or "none", "->", png)
