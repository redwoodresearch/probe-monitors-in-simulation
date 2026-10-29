"""Write-up figure eb4bbb7c, "Reply-reading detector breaks on formatting; command-reading doesn't" (FIGURES.md on
aswin/writeup-numbers-sheet @ af47bc10, outline lines 35 and 132). $0, local, one small JSON.

Source: experiments/2026-10-08_q3_swap_cue/results/fix939_POSTHOC_2026-10-11_020519.json, the S3 original listed in
results/manifest_phase1_checks_fix939_v1.json (sha256 checked here). Every drawn value is read by key; each is printed
with its key path and compared with the number on the doc image (MATCH / MISMATCH at the image's one decimal).
Bars: simulated minus real catch at the 1% false-alarm target (attack detector #36, thresholds frozen on real honest
runs), 939 cells over 536 main tasks: as simulated, after the two formatting fixes, and two real runs (run 2 - run 1).
Readouts: reply = output_end (layer 63); command = call_end (layer 48, all calls), as analyze_fix939.py scored them.

  uv run --no-project --with matplotlib --with numpy python experiments/2026-10-12_q3_writeup_figures/plot_eb4bbb7c.py \
      --src <fix939_POSTHOC_2026-10-11_020519.json downloaded from S3>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parent.parent
MANIFEST = REPO / "experiments/2026-10-08_q3_swap_cue/results/manifest_phase1_checks_fix939_v1.json"
NAME = "fix939_POSTHOC_2026-10-11_020519.json"
RATE = "0.01"
# (readout key, cell key, bar label, number on the doc image)
BARS = [
    ("output_end", "as_is_sim_minus_e2", "As simulated: sim \u2212 real", -10.4),
    ("output_end", "fixed_sim_minus_e2 (primary)", "After two formatting fixes: sim \u2212 real", +0.9),
    ("output_end", "real_e2_minus_e1 (e1 was in training)", "Real run 2 \u2212 real run 1", -0.9),
    ("call_end", "as_is_sim_minus_e2", "As simulated: sim \u2212 real", +0.6),
    ("call_end", "fixed_sim_minus_e2 (primary)", "After two formatting fixes: sim \u2212 real", -0.5),
    ("call_end", "real_e2_minus_e1 (e1 was in training)", "Real run 2 \u2212 real run 1", +2.3),
]
DOC_TITLE = "Reply-reading detector breaks on formatting; command-reading doesn't"  # the doc image's title, not reused
TITLE = "Catch differences before and after two formatting fixes (frozen 1% thresholds)"  # descriptive (coordinator; #1024 review)
COLORS = ["#c4661a", "#1f8a70", "#2f6eb5"]  # by position within a group: as simulated, after fixes, real run 2 - run 1
GROUPS = {"output_end": "Reply end (layer 63)", "call_end": "Call end (layer 48; all calls)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    want = next(f["sha256"] for f in json.loads(MANIFEST.read_text())["files"] if f["file"] == NAME)
    got = hashlib.sha256(Path(args.src).read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{args.src}: sha256 {got} != manifest {want}")
    d = json.loads(Path(args.src).read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()

    rows, n_mismatch = [], 0
    for ro, cell, label, doc in BARS:
        r = d["readouts"][ro]
        e = r[RATE][cell]
        v, (lo, hi) = 100 * e["observed"], (100 * e["ci95"][0], 100 * e["ci95"][1])
        ok = round(v, 1) == doc
        n_mismatch += not ok
        key = f'readouts.{ro}."{RATE}"."{cell}"'
        print(f"{key}: {v:+.2f} [{lo:+.2f}, {hi:+.2f}]  doc {doc:+.1f}  {'MATCH' if ok else 'MISMATCH'}")
        rows.append({"key": key, "readout": ro, "label": label.replace("\n", " "), "value": v, "ci95": [lo, hi],
                     "doc": doc, "match": ok, "cells": r["cells"], "tasks": r["tasks"]})
    cells = {r["cells"] for r in rows}
    tasks = {r["tasks"] for r in rows}
    if len(cells) != 1 or len(tasks) != 1:
        raise SystemExit(f"cells or tasks differ across readouts: {cells} {tasks}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    y, ticks, ticklabels, bold = 0.0, [], [], []
    for gi, ro in enumerate(GROUPS):
        ticks.append(y)  # the group header is its own tick row, so it can never be clipped by the axes
        ticklabels.append(GROUPS[ro])
        bold.append(True)
        y += 0.8
        for j, row in enumerate(r for r in rows if r["readout"] == ro):
            v, (lo, hi) = row["value"], row["ci95"]
            ax.barh(y, v, 0.62, color=COLORS[j])
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="black", capsize=3, lw=1.1)
            if v < -2:  # a long negative bar: put the label left of its interval so it doesn't sit on the bar
                ax.text(lo - 0.5, y, f"{v:+.1f}  [{lo:+.1f}, {hi:+.1f}]", va="center", ha="right", fontsize=9)
            else:
                ax.text(hi + 0.5, y, f"{v:+.1f}  [{lo:+.1f}, {hi:+.1f}]", va="center", ha="left", fontsize=9)
            ticks.append(y)
            ticklabels.append(row["label"])
            bold.append(False)
            y += 0.8
        y += 0.9
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(ticks, ticklabels, fontsize=9)
    for lab, b in zip(ax.get_yticklabels(), bold):
        if b:
            lab.set_fontweight("bold")
            lab.set_fontsize(10)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(-31, 14)
    ax.set_xlabel("Catch difference at a 1% false-alarm target (points; 0 = no gap, closer to 0 is better)", fontsize=8.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title(TITLE, fontsize=10.5, pad=12)
    n_c, n_t = cells.pop(), tasks.pop()
    import textwrap

    foot = [f"{n_c} matched cells, {n_t} tasks; points at thresholds frozen on real honest runs; lines: 95% task-bootstrap intervals, "
            "thresholds re-estimated per draw. Run 1 was in the detector's training.",
            "Two fixes (fix939): insert outputs gain the real review note and lose the trailing newline; str_replace diffs lose the "
            "trailing newline, and plain diffs gain the real numbered gutter.",
            "Reply end only, with each world's own threshold (matched-false-alarm check): the 5% after-fix gap is not readable; "
            "1% is reported without a verdict."]
    fig.text(0.01, 0.01, "\n".join(textwrap.fill(s, 150) for s in foot), fontsize=7, color="#5a6371")
    fig.tight_layout(rect=(0, 0.15, 1, 1))

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    png = HERE / f"fig_eb4bbb7c_{stamp}.png"
    with png.open("xb") as fh:
        fig.savefig(fh, dpi=170, format="png")
    out = {"figure": "eb4bbb7c", "title": TITLE, "doc_title": DOC_TITLE,
           "source": {"file": NAME, "sha256": got, "manifest": str(MANIFEST.relative_to(REPO))}, "rate": RATE, "git": head,
           "png": png.name, "bars": rows, "mismatches": n_mismatch}
    with (HERE / f"values_eb4bbb7c_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(f"wrote {png.name}; mismatches vs doc image: {n_mismatch}")


if __name__ == "__main__":
    main()
