# 2026-10-12_q3_sabotage_learning

2026-10-12 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

How does the stage-1 sabotage probe's accuracy grow with the number of labelled training tasks, and does it beat label-shuffle nulls and seeded plants?

## What we did

Learning curves (6, 12, 24 training tasks) for two readouts (call end, command end) on Qwen3.5-27B activations on Modal (2,542 fits, \$0.23), plus fixed-setting and selection-aware label-shuffle nulls and seeded plant checks.

## What we found

README: AUROC rises from 0.854 to 0.910 (call end) and 0.847 to 0.876 (command end) from 6 to 24 tasks; catch from 13% to 25% and 17% to 37%; every level beats every selection-aware null scramble.

## In the post?

Not in the post.

## Status

Exploratory (post hoc); scaled up in [2026-10-13_q3_sabotage_scaleup](../2026-10-13_q3_sabotage_scaleup/)

## Builds on

[2026-10-11_sabotage_stage1_labels](../2026-10-11_sabotage_stage1_labels/)

## Main scripts

- [`run_learning.py`](run_learning.py)
- [`analyze_learning.py`](analyze_learning.py)
- [`run_null_shuffle.py`](run_null_shuffle.py)
- [`run_selaware_null.py`](run_selaware_null.py)
- [`build_rows_command.py`](build_rows_command.py)

Result files the index names:

- `learning_result_2026-10-12_144211.json` (not in this folder at the source commit)
- `selaware_null_summary_2026-10-13_111302.json` (not copied: result left out to keep the release under 120 MB)
- `null_shuffle_summary_2026-10-13_102819.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_sabotage_learning/` at commit `ddd79ec04c3acc4e9d635cb4475419f7d392958a` of our internal repository. 28 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
