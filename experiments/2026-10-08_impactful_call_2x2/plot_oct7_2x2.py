"""The Oct 7 Q2 2x2 figure, redrawn with 95% intervals from the final table (no new compute).

Design as on Oct 7: Qwen3-32B, attack runs, all eligible calls, threshold frozen at 1% false alarms on the training
world's validation calls, final labeler eb014307fd1aab31. Source:
../2026-10-06_q2_impactful_call_transfer/results/q2_final_table_2026-10-07_095958_056930.md (95% intervals resample
main tasks). Layout as on Oct 7: one 2x2 per readout, grouped by training world; grey = tested on the same kind of run,
blue = switched; the finished command (call_end) left, before the command (pre_call) right, with the command-text
baseline as a line. pre_call is a template token, so its content-token control (prev_content) is drawn beside it.

  uv run --with matplotlib python experiments/2026-10-08_impactful_call_2x2/plot_oct7_2x2.py [--finished-only]
--finished-only: the meeting variant, the finished-command panel alone (both before-command panels left out together,
so the readout rule still holds).
"""
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
TABLE = HERE.parents[0] / "2026-10-06_q2_impactful_call_transfer" / "results" / "q2_final_table_2026-10-07_095958_056930.md"
ROW = re.compile(r"^\| (call_end|pre_call|prev_content|baseline: command_text) \| (real|sim) → (real|sim) \| "
                 r"([\d.]+) \[([\d.]+), ([\d.]+)\] \| ([\d.]+)% \[([\d.]+), ([\d.]+)\] \|", re.M)
PANELS = [("call_end", "The finished command"), ("pre_call", "Before the command is written"),
          ("prev_content", "Before the command: last text token\n(control for the panel to its left)")]
GREY, BLUE = "#9e9e9e", "#1f77b4"
NAMES = {"real": "real", "sim": "simulated"}


def parse():
    """The first 2x2 block of the final table (all eligible calls): readout -> (train, test) -> numbers."""
    text = TABLE.read_text()
    block = text.split("## 2×2, all eligible calls", 1)[1].split("\n## ", 1)[0]
    cells = {}
    for m in ROW.finditer(block):
        ro, tr, te = m.group(1), m.group(2), m.group(3)
        cells.setdefault(ro, {})[(tr, te)] = {"catch": float(m.group(4)), "ci": [float(m.group(5)), float(m.group(6))],
                                              "false_alarms": float(m.group(7)) / 100, "fa_ci": [float(m.group(8)) / 100, float(m.group(9)) / 100]}
    for ro in ("call_end", "pre_call", "prev_content", "baseline: command_text"):
        if len(cells.get(ro, {})) != 4:
            raise ValueError(f"{TABLE.name}: expected 4 cells for {ro}, found {len(cells.get(ro, {}))}")
    return cells


def main():
    cells = parse()
    finished_only = "--finished-only" in sys.argv
    panels = PANELS[:1] if finished_only else PANELS
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 if finished_only else 15.5, 6.2 if finished_only else 5.6), squeeze=False)
    axes = axes[0]
    for ax, (ro, title) in zip(axes, panels):
        order = [("real", "real"), ("real", "sim"), ("sim", "sim"), ("sim", "real")]
        xs = [0, 0.42, 1.2, 1.62]
        for x, (tr, te) in zip(xs, order):
            c = cells[ro][(tr, te)]
            ax.bar(x, c["catch"], 0.38, color=GREY if tr == te else BLUE,
                   yerr=[[c["catch"] - c["ci"][0]], [c["ci"][1] - c["catch"]]], capsize=4)
            label = f"{c['catch']:.1%}\n{c['false_alarms']:.1%} false\nalarms"
            if c["catch"] > 0.5:  # tall bars: inside, so the labels stay clear of the baseline line
                ax.text(x, c["ci"][0] - 0.03, label, ha="center", va="top", fontsize=8, color="white")
            else:
                ax.text(x, c["ci"][1] + 0.02, label, ha="center", va="bottom", fontsize=8)
            ax.text(x, -0.03, f"tested on\n{NAMES[te]}", ha="center", va="top", fontsize=8)
        if ro == "call_end":
            b = cells["baseline: command_text"][("real", "real")]["catch"]
            ax.axhline(b, color="k", ls="--", lw=1, zorder=0)
        ax.set_xticks([0.21, 1.41], ["trained on real runs", "trained on simulated runs"])
        ax.tick_params(axis="x", pad=28)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("Impactful commands caught (higher is better)")
    b = cells["baseline: command_text"][("real", "real")]["catch"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Line2D([], [], color="k", ls="--", lw=1)]
    fig.legend(handles, ["tested on the same kind of run it was trained on", "switched: tested on the other kind",
                         f"{'' if finished_only else 'finished command: '}command words alone (text classifier, real → real) {b:.0%}"],
               loc="upper center", bbox_to_anchor=(0.5, 0.88 if finished_only else 0.905), ncol=1 if finished_only else 3, fontsize=9, frameon=False)
    sub = "Off-policy: Qwen3-32B reads Sol's transcripts." + ("\n" if finished_only else " ") + "Same figure as Oct 7, now with 95% intervals."
    fig.suptitle("Does the impactful-command probe work the same on simulated runs?\n" + sub, fontsize=12)
    import textwrap
    caption = ("Caught = an impactful command (one that changes something) scores above a threshold set so the probe flags "
               "1% of read-only commands in the world it was trained on; the false-alarm rate it then gives on the test world is "
               "printed on each bar. Attack runs, all commands. Error bars: 95% intervals from resampling tasks. Source: "
               f"{TABLE.name}.")
    if finished_only:
        caption += (" Before-command readouts omitted (manager's focus is the finished command); see the 3-panel version.")
    fig.text(0.5, 0.01, "\n".join(textwrap.wrap(caption, 95 if finished_only else 200)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.13 if finished_only else 0.06, 1, 0.78 if finished_only else 0.86))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    png = HERE / "results" / f"oct7_2x2_with_intervals{'_finished_only' if finished_only else ''}_{stamp}.png"
    fig.savefig(png, dpi=150)
    png.with_suffix(".json").write_text(json.dumps({"source": str(TABLE.relative_to(HERE.parents[1])),
                                                    "cells": {ro: {f"{a}->{b}": v for (a, b), v in c.items()} for ro, c in cells.items()}}, indent=1))
    print(png)


if __name__ == "__main__":
    main()
