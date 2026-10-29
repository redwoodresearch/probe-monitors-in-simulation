"""Figure 1: honest main-task success per acting model, from analyze_fig1_honest_success.py's result. Bars with exact
95% intervals; errored, not-scored and missing attempts are listed under each bar, never counted as failures.

  uv run --no-project --python 3.12 --with matplotlib python plot_fig1.py results/fig1_honest_success_<stamp>.json [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = json.loads(Path(sys.argv[1]).read_text())
models = list(R["rates"])
fig, ax = plt.subplots(figsize=(max(4.2, 1.8 + 1.6 * len(models)), 4.0))
for i, m in enumerate(models):
    r, c = R["rates"][m], R["counts"][m]
    lo, hi = r["exact_95"]
    ax.bar(i, r["rate"], color="#4C72B0", width=0.6)
    ax.errorbar(i, r["rate"], yerr=[[r["rate"] - lo], [hi - r["rate"]]], color="black", capsize=4)
    ax.text(i, hi + 0.02, f"{r['successes']}/{r['scored']} = {r['rate']:.0%}", ha="center", va="bottom", fontsize=9)
    extra = ", ".join(f"{k} {c.get(k, 0)}" for k in ("errored", "not scored", "missing") if c.get(k))
    ax.text(i, -0.12, extra or "no errored or unscored runs", ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
ax.set_xticks(range(len(models)), [m.split("/")[-1] for m in models])
ax.set_ylim(0, 1.12)
ax.set_ylabel("main-task success\n(higher is better)")
title = "Honest main-task success per acting model\n(exact 95% intervals; errored and unscored listed, not counted)"
ax.set_title(f"{R['label'] + ' — ' if R['label'] else ''}{title}", fontsize=8)
fig.tight_layout()
out = Path(sys.argv[1]).with_suffix(".png").name.replace("fig1_honest_success", "figure1")
path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "figures" / out  # optional: a live copy elsewhere
path.parent.mkdir(exist_ok=True)
fig.savefig(path, dpi=150)
print("->", path)
