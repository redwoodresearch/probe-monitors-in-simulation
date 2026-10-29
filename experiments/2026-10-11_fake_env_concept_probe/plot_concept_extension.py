"""Plot stage 5's pre-registered reading beside the post hoc extension (README "Extension"): R and Δc per readout.

  uv run --no-project --with matplotlib python experiments/2026-10-11_fake_env_concept_probe/plot_concept_extension.py --real results/concept_real_analysis_<stamp>.json --extension results/concept_real_analysis_extension_<stamp>.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--real", required=True)
ap.add_argument("--extension", required=True)
a = ap.parse_args()
pre, ext = (json.loads((HERE / f).read_text()) for f in (a.real, a.extension))
if not ext["reading"].startswith("post hoc extension"):
    raise ValueError("--extension is not an extension analysis")
K, D = "R = mean|sim - e1| / mean|e2 - e1|", "dc = P(sim > e1) - P(e2 > e1)"
runs = [(pre, f"pre-registered\n({pre['cells in the main comparison']} cells)", "#9AA5B1"),
        (ext, f"with the extension\n({ext['cells in the main comparison']} cells, post hoc)", "#4C72B0")]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
ax.axhline(1.2, color="#777777", lw=0.8, ls="--")
ax.text(1.62, 1.2, "1.2 margin", fontsize=7, va="bottom", ha="right", color="#555555")
ax.axhline(1, color="black", lw=0.8)
bx.axhspan(-0.05, 0.05, color="#E5E5E5")
bx.axhline(0, color="black", lw=0.8)
w = 0.2
for j, (r, name, col) in enumerate(runs):
    for i, k in enumerate(("C", "T")):
        for axis, key, ci, fmt in ((ax, K, "R 95%", "{:.2f}"), (bx, D, "dc 95%", "{:+.3f}")):
            v, (lo, hi) = r[k][key], r[k][ci]
            x = i + (j - 0.5) * w * 2
            axis.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="o", ms=7, capsize=5, color=col, label=name if (i == 0 and axis is ax) else None)
            axis.annotate(fmt.format(v), (x, v), textcoords="offset points", xytext=(9, -3), fontsize=9)
for axis in (ax, bx):
    axis.set_xticks(range(2), ["content (decides)", "turn start"], fontsize=9)
    axis.set_xlim(-0.6, 1.7)
ax.set_ylabel("real-vs-sim gap ÷ real-vs-real gap\n(whole interval above 1.2 = sim differs more)")
bx.set_ylabel("P(sim > real) − P(real 2 > real)\n(above 0 = sim reads more 'fake')")
bx.text(1.62, 0.05, "±0.05", fontsize=7, va="bottom", ha="right", color="#555555")
fig.legend(fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.suptitle(f"Content readout: {pre['C']['verdict (deciding)']} on {pre['cells in the main comparison']} cells; "
             f"{ext['C']['verdict (deciding)']} on {ext['cells in the main comparison']}", fontsize=10)
fig.text(0.5, -0.1, "95% intervals: 2,000 draws of main tasks. The extension's cells were added after the pre-registered result.",
         ha="center", fontsize=8)
fig.tight_layout()
out = HERE / "results" / f"concept_extension_figure_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"-> {out}")
