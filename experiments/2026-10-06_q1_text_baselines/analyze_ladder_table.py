"""The scrub ladder as one readable Markdown file: a summary table (steps as rows) and each step's top snippets.

Reads a tfidf_ladder summary from results/ and writes results/tfidf_ladder_table_<stamp>.md.

  uv run python experiments/2026-10-06_q1_text_baselines/analyze_ladder_table.py tfidf_ladder_summary_<stamp>.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
STEPS = {"ladder_step0": "0: as the probe reads it", "ladder_step1": "1: + times and dates masked",
         "ladder_step2": "2: + numbers, hex IDs, UUIDs, IPs masked", "ladder_step3": "3: + whitespace normalized"}
SHOW = {"\n": "⏎", "\t": "⇥", "\r": "␍"}


def fmt(r):
    return f"{r['outputs']['auroc']:.3f} / {r['runs']['auroc']:.3f} / {r['pairs']['accuracy']:.3f}"


def snippet(grams):
    return " / ".join("`" + "".join(SHOW.get(c, c) for c in g).replace("|", "\\|").replace("`", "'") + "`" for g in grams) or "?"


def main():
    path = HERE / "results" / Path(sys.argv[1]).name
    s = json.loads(path.read_text())
    b = s["baselines"]
    lines = [f"# Scrub ladder: which tells carry real vs simulated? ({path.name})", "",
             ("TF-IDF (cleaned text) on the 939 v3 cells, out-of-fold on the shared folds. Each step adds its scrub to the "
              + "steps before it, applied to every arm. Cells: AUROC per output / AUROC per run / pairwise accuracy (share of "
              + "cells whose simulated run scores above its real run). The e1-vs-e2 null should be 0.5. Step 4 is step 3 "
              + "with editor outputs dropped."), "",
             "| step | real vs sim, all outputs | real vs sim, no editor outputs | null e1 vs e2, all | null e1 vs e2, no editor |",
             "|---|---|---|---|---|"]
    for k, label in STEPS.items():
        r = b[k]["results"]
        lines.append(f"| {label} | {fmt(r['real_vs_sim/all']['all_cells'])} | {fmt(r['real_vs_sim/no_editor']['all_cells'])} | "
                     f"{fmt(r['null_e1_vs_e2/all']['all_cells'])} | {fmt(r['null_e1_vs_e2/no_editor']['all_cells'])} |")
    r = b["ladder_step3"]["results"]
    lines += [f"| 4: step 3, editor outputs dropped | {fmt(r['real_vs_sim/no_editor']['all_cells'])} | | "
              f"{fmt(r['null_e1_vs_e2/no_editor']['all_cells'])} | |", "",
              "95% intervals (bootstrap over main tasks) are in the summary file.", ""]
    for k, label in STEPS.items():
        top = b[k]["top_features"]["real_vs_sim"]
        n = len(top["toward_sim-attack"])
        lines += [f"## Step {label}: top {n} snippets, real epoch 1 vs simulated (all outputs)", "",
                  "| # | toward simulated | weight | toward real | weight |", "|---|---|---|---|---|"]
        for i, (a, c) in enumerate(zip(top["toward_sim-attack"], top["toward_real-attack-e1"]), 1):
            lines.append(f"| {i} | {snippet(a['ngrams'])} | {a['weight']:+.2f} | {snippet(c['ngrams'])} | {c['weight']:+.2f} |")
        lines.append("")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"tfidf_ladder_table_{stamp}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
