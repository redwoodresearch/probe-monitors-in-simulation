# 2026-10-13_q3_sabotage_avg_repair

2026-10-13 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Do the sabotage-probe training-grid results (probes that flag the sabotage command itself on Qwen3.5-27B's own runs) change with a command readout averaged over tokens, or when Qwen3.5-27B's simulated runs are repaired with the formatting fixes?

## What we did

Part 1 extracted Qwen3.5-27B layer-30 means over each command's tokens and reran the sabotage scale-up's training grid (real-trained, simulation-trained, mixed); part 2 applied all formatting fixes except the clock fix inside the logged token ids of Qwen3.5-27B's September simulated runs, re-extracted, and reran the grid at last-token and averaged readouts. Modal.

## What we found

codex/03_b2_results.md: the repair moves almost nothing (largest simulated-catch change anywhere in the grid 1.24 points; frozen gate simulated catch 16.7% before and after, simulated minus first real -4.2 [-12.1, +3.1]); results for the command readout averaged over tokens on S3 (posthoc_b1_matched_2026-10-13_171525.json).

## In the post?

Not in the post.

## Status

Exploratory (labels not audited)

## Builds on

[2026-10-13_q3_sabotage_scaleup](../2026-10-13_q3_sabotage_scaleup/) (grid, inputs); [2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/) (fixes)

## Main scripts

- [`run_extract_command_mean.py`](run_extract_command_mean.py)
- [`run_grid_avg.py`](run_grid_avg.py)
- [`analyze_grid.py`](analyze_grid.py)
- [`analyze_posthoc_b1.py`](analyze_posthoc_b1.py)
- [`analyze_posthoc_b1_matched.py`](analyze_posthoc_b1_matched.py)
- [`stage_b2.py`](stage_b2.py)
- [`run_extract_b2.py`](run_extract_b2.py)
- [`run_grid_b2.py`](run_grid_b2.py)

Result files the index names:

- `posthoc_b1_matched_2026-10-13_171525.json` (not copied: result left out to keep the release under 120 MB)
- [`stage_b2_stage_2026-10-13_170814.json`](results/stage_b2_stage_2026-10-13_170814.json)
- `manifest_b1_grid_s3_v2.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_b2_grid_s3_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

16 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_q3_sabotage_avg_repair/` at commit `ab421676ed0e8370a0c8ae3caa0edcd68e2132e6` of our internal repository. 20 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
