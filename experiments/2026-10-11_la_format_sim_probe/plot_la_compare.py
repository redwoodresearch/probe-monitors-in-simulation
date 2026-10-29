"""N1 Part 1's figure: N1 minus the concept probe, in held-out AUROC, on identical LinuxArena-format pairs.

  uv run --no-project --with matplotlib python experiments/2026-10-11_la_format_sim_probe/plot_la_compare.py results/compare_<stamp>.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.loads((HERE / sys.argv[1]).read_text())
bars = [("content (decides)\nhonest held-out", d["honest held-out implied"], "#4C72B0"),
        ("content (decides)\nattack panel", d["attack panel implied"], "#4C72B0"),
        ("turn start\nhonest held-out", d["turn start (N1 at 24, concept at 16), honest held-out implied"], "#9AA5B1"),
        ("turn start\nattack panel", d["turn start, attack panel implied"], "#9AA5B1")]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.axhline(0, color="black", lw=0.8)
ax.axhline(0.05, color="#777777", lw=0.8, ls="--")
ax.text(3.45, 0.05, "+0.05 bar", fontsize=8, va="bottom", ha="right", color="#555555")
for i, (name, r, col) in enumerate(bars):
    v, (lo, hi) = r["difference"], r["difference 95%"]
    b = ax.bar(i, v, 0.6, yerr=[[v - lo], [hi - v]], capsize=4, color=col)
    ax.bar_label(b, labels=[f"{v:+.3f}"], padding=3, fontsize=9)
ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8)
ax.set_ylabel("N1 minus concept probe, held-out AUROC\n(higher = LinuxArena training transfers better)")
ax.set_title("Training in LinuxArena format adds about +0.02 AUROC on the deciding content readout", fontsize=10)
fig.text(0.5, -0.06, "Implied cues, identical pairs. 95% intervals: 2,000 draws of main tasks. "
         "Reading an inserted cue isn't the same as the model noticing it's in a simulation.", ha="center", fontsize=7.5)
fig.tight_layout()
out = HERE / "results" / f"compare_figure_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.png"
if out.exists():  # results/ is append-only
    raise FileExistsError(out)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"-> {out}")
