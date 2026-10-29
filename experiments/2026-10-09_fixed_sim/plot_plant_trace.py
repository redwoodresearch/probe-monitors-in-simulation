"""Figure for planted check 1, a closer look (exploratory): per-token scores of step A's real-vs-sim probe and the aimed
probe, for 5 seeded pairs (planted copy and its twin), around the output that holds the planted line. Reads
results/plant_trace_<stamp>.json (numbers only); writes results/plant_trace_<stamp>.png.

  uv run --no-project --with matplotlib --with numpy python experiments/2026-10-09_fixed_sim/plot_plant_trace.py results/plant_trace_<stamp>.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(path, half: int = 120):
    d = json.loads(Path(path).read_text())
    tr = d["traces"]
    fig, ax = plt.subplots(len(tr), 2, figsize=(12, 2.0 * len(tr)), sharey="col", sharex=True)
    for i, t in enumerate(tr):
        p, w = t["planted"], t["twin"]
        line = np.flatnonzero(p["line"])
        l0, n_line = line[0], len(line)
        x_p = np.arange(len(p["output"])) - l0  # 0 = the line's first token
        x_w = np.arange(len(w["output"])) - l0
        x_w = np.where(x_w >= 0, x_w + n_line, x_w)  # the twin lacks the line: its later tokens line up after it
        out = np.array(p["output"])
        site_end_p = np.flatnonzero(out == 0)[-1] - l0  # the planted output's end-of-output readout (step A's)
        site_end_w = np.flatnonzero(np.array(w["output"]) == 0)[-1] - l0 + n_line
        for j, (probe, title) in enumerate((("real_vs_sim", "Step A's real-vs-sim probe (layer 63)"),
                                            ("aimed", "Probe aimed at the planted line (layer 63)"))):
            a = ax[i, j]
            a.axvspan(-0.5, n_line - 0.5, color="#f1c40f", alpha=0.5, lw=0, label="planted line")
            for k in np.flatnonzero(np.diff(out)):  # output ends
                a.axvline(x_p[k] + 0.5, color="0.75", lw=0.7, ls="--")
            # break the twin's line at the planted line's position (NaN), so it is not drawn across the gap it lacks
            a.plot(np.insert(x_w.astype(float), l0, np.nan), np.insert(np.asarray(w[probe], float), l0, np.nan), color="0.5", lw=0.9,
                   label="twin (no line)")
            a.plot(x_p, p[probe], color="#c0392b", lw=0.9, label="planted copy")
            a.plot([site_end_w], [w[probe][site_end_w - n_line + l0]], "o", color="0.4", ms=5)
            a.plot([site_end_p], [p[probe][site_end_p + l0]], "o", color="#c0392b", ms=5, label="end-of-output readout")
            a.set_xlim(-half, n_line + half)
            if i == 0:
                a.set_title(title, fontsize=11)
            if j == 0:
                a.set_ylabel(f"pair {i + 1}\n{t['env']}", fontsize=9)
            if i == len(tr) - 1:
                a.set_xlabel("tokens from the planted line (dashed: output ends)", fontsize=9)
            a.tick_params(labelsize=8)
    ax[0, 1].legend(fontsize=8, loc="upper right", frameon=False)
    fig.text(0.01, 0.995, "left: higher = more 'simulated'", fontsize=9, va="top")
    fig.text(0.51, 0.995, "right: higher = 'planted line'", fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out_png = Path(path).with_name(f"{Path(path).stem}_plot_{stamp}.png")  # a new file each time; earlier figures are kept
    fig.savefig(out_png, dpi=130)
    print("->", out_png)


if __name__ == "__main__":
    main(sys.argv[1])
