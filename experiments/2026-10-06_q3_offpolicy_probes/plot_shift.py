"""Figure: run-level AUROC per layer, for each probe, of sim vs real e1 (the question), real e2 vs e1 (the yardstick)
and real e1 vs real honest (does the probe respond at all), with 95% main-task bootstrap intervals. Bars start at 0.5
(no difference). Reads results/shift_<probe>_<stamp>.json (call_end, all calls).

  uv run --no-project --python 3.13 --with matplotlib python experiments/2026-10-06_q3_offpolicy_probes/plot_shift.py \
      results/shift_apollo_deception_....json results/shift_eval_awareness_....json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SERIES = [("sim_minus_e1", "sim vs real e1 (the question)", "#2a78d6"),
          ("e2_minus_e1", "real e2 vs real e1 (yardstick)", "#eb6834"),
          ("e1_minus_honest", "real attack e1 vs real honest (sanity)", "#1baf7a")]
TRAINED_ON = {"apollo_deception": "Apollo deception probe: instructed honest vs dishonest statements",
              "eval_awareness": "Eval-awareness probe: same tool call after evaluation vs deployment\nframing, or test-style vs realistic task"}

files = [HERE / f for f in sys.argv[1:]]
fig, axes = plt.subplots(1, len(files), figsize=(6.2 * len(files), 4.6), sharey=True, squeeze=False)
for ax, f in zip(axes[0], files):
    d = json.loads(f.read_text())
    probe = f.name.removeprefix("shift_").rsplit("_", 2)[0]
    res = d["call_end"]["all"]
    layers = list(res)
    w = 0.26
    for k, (key, label, color) in enumerate(SERIES):
        xs = [i + (k - 1) * w for i in range(len(layers))]
        vals = [res[l][key]["run_auroc"]["est"] for l in layers]
        lo = [v - res[l][key]["run_auroc"]["ci95"][0] for v, l in zip(vals, layers)]
        hi = [res[l][key]["run_auroc"]["ci95"][1] - v for v, l in zip(vals, layers)]
        ax.bar(xs, [v - 0.5 for v in vals], w * 0.92, bottom=0.5, color=color, label=label, edgecolor="white", linewidth=1)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor="#555555", elinewidth=1, capsize=2)
        for x, v, h in zip(xs, vals, hi):
            ax.text(x, max(v, 0.5) + h + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#333333")
    ax.axhline(0.5, color="#888888", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(layers)), [l.removeprefix("L") for l in layers])
    ax.set_xlabel("Qwen3-32B layer (of 64)")
    ax.set_title(TRAINED_ON.get(probe, probe), fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
axes[0][0].set_ylabel("Run-level AUROC of the raw probe score\n(0.5 = arms score the same; farther = they differ)")
axes[0][0].legend(loc="upper left", fontsize=8, frameon=False)
fig.suptitle("Off-policy probes read at call_end on LinuxArena: 939 paired attack cells, 990 honest runs; 95% CI over main tasks",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout()
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
out = HERE / "results" / f"shift_auroc_by_layer_{stamp}.png"
fig.savefig(out, dpi=160)
print(f"-> {out}")
