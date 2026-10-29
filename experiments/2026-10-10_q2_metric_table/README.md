# 2026-10-10_q2_metric_table

2026-10-10 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

What do every command and deception monitor's results look like on both measures (AUROC and catch at 1% and 5% false alarms), and does the destructive gap survive a corrected-label refit with separate calibration?

## What we did

Post hoc table from stored scores, then a separate-calibration refit of the destructive probe on the corrected label rule and of the transfer probe, plus a selection-aware shuffled-label null (20 shuffles).

## What we found

Destructive gap +3.8 points at 1%, transfer +6.8 at 1%, run-level deception -1.2 [-7.2, +4.2]. Corrected destructive gap +2.8 [-3.1, +9.7] at 1%, which does not confirm the earlier +10.5 at 5%. Real 0.920 and simulated 0.924 beat all 20 shuffles.

## In the post?

Not in the post.

## Status

Current (post hoc, preliminary)

## Builds on

[2026-10-10_q2_destructive_commands](../2026-10-10_q2_destructive_commands/); [2026-10-10_q2_transfer_commands](../2026-10-10_q2_transfer_commands/); [2026-10-10_q2_deception_runlevel](../2026-10-10_q2_deception_runlevel/)

## Main scripts

- [`posthoc_metrics.py`](posthoc_metrics.py)
- [`run_separate_calibration.py`](run_separate_calibration.py)
- [`analyze_separate_calibration.py`](analyze_separate_calibration.py)
- [`null_separate_calibration.py`](null_separate_calibration.py)
- [`run_selection_null.py`](run_selection_null.py)
- [`analyze_selection_null.py`](analyze_selection_null.py)
- [`build_recal_labels.py`](build_recal_labels.py)

Result files the index names:

- `metric_table_2026-10-10_224321.json` (not copied: result left out to keep the release under 120 MB)
- [`separate_calibration_2026-10-10_225753.json`](results/separate_calibration_2026-10-10_225753.json)
- `separate_calibration_null_2026-10-10_225753.json` (not copied: result left out to keep the release under 120 MB)
- [`selection_null_2026-10-10_232652.json`](results/selection_null_2026-10-10_232652.json)

## Files here

7 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-10_q2_metric_table/` at commit `7aad630049c5ee4b10f6a7c7b5158ae2bd4ea8d3` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
