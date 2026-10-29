"""Claim B's table (PLAN.md amendment 2, "Output"): the four comparisons (fixed sim vs real e1 and real e2 vs real e1;
raw and times masked) and the two planted checks, TF-IDF beside the probe, from the result files only:
  TF-IDF:  results/gate_a_2026-10-09_201142.json (leave one environment out; null 200 draws, per main task)
  probe A: results/claimB_probe_summary_2026-10-10_021700.json (output_end; d9 a1821be0; null per cell, see PLAN.md)
  probe B: results/claimB_b_grid_2026-10-10_050016_d9-857cb3cc.json (each readout, and the selection over all four;
           d9 857cb3cc; null per main task / per planted pair)
All on the 5 test environments; probe numbers are the nested selection-aware headline (best validation layer and C).
Writes results/claimB_table_<stamp>.md.

  uv run python experiments/2026-10-09_fixed_sim/plot_claimB_table.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
R = HERE / "results"
TFIDF = json.loads((R / "gate_a_2026-10-09_201142.json").read_text())["rows"]
A = json.loads((R / "claimB_probe_summary_2026-10-10_021700.json").read_text())["levels"]
B = json.loads((R / "claimB_b_grid_2026-10-10_050016_d9-857cb3cc.json").read_text())["levels"]
ROWS = [("fixed sim vs real e1, raw", "fixed_sim_vs_e1|loeo|all", "level0", "fixed_vs_e1"),
        ("fixed sim vs real e1, times masked", "fixed_sim_vs_e1|loeo|level1", "level1", "fixed_vs_e1"),
        ("real e2 vs real e1, raw", "e2_vs_e1|loeo|all", "level0", "e2_vs_e1"),
        ("real e2 vs real e1, times masked", "e2_vs_e1|loeo|level1", "level1", "e2_vs_e1"),
        ("planted check 1, explicit line (bar 0.95)", None, "level1", "plant1"),
        ("planted check 2, R2 undone (bar 0.90)", None, "level1", "plant2")]
READOUTS = ["output_end", "output_end_ema0.8", "output_end_ema0.95", "output_end_mean"]


def fmt(v) -> str:
    return f"{v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"


def b_entry(level, comp, readout):
    by = B[level][comp]
    node = by["by_readout"][readout] if readout != "all" else by["all_readouts"]
    v = node["best"]["test_envs"]
    return f"{fmt(v['test'])}, null p95 {v['null']['null_p95']:.3f}"


def main():
    lines = ["| comparison | TF-IDF | probe A, end of output | probe B, end of output | B, moving avg 0.8 | B, moving avg 0.95 | B, mean | B, chosen over all 4 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, tkey, level, comp in ROWS:
        if tkey:
            t = TFIDF[tkey]
            tf = f"{t['test_envs']['auroc']:.3f} [{t['test_envs']['ci95_env_resample'][0]:.3f}, {t['test_envs']['ci95_env_resample'][1]:.3f}]"
            if "null" in t and t["null"].get("range"):
                tf += f", null [{t['null']['range'][0]:.3f}, {t['null']['range'][1]:.3f}]"
        else:
            tf = "(not run)"
        a = A[level][comp]["best_layer_and_C"]["test_envs"]
        cells = [tf, f"{fmt(a['test'])}, null p95 {a['null']['null_p95']:.3f}"] + [b_entry(level, comp, r) for r in READOUTS + ["all"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = R / f"claimB_table_{stamp}.md"
    head = ("Run-level AUROC [95% interval] on the 5 test environments. TF-IDF interval: resampling environments; probe "
            "intervals: resampling main tasks. Probe A's null swapped per cell (PLAN.md caveat); B's per main task or planted pair.\n\n")
    with open(path, "x") as f:
        f.write(head + "\n".join(lines) + "\n")
    print(head + "\n".join(lines))
    print(path)


if __name__ == "__main__":
    main()
