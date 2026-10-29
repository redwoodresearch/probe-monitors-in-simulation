# 2026-10-14_simhonest_a25_09

2026-10-14 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

If simulated attacks are thresholded on simulated honest runs, as a fully simulated evaluation would do, how far is simulated catch from real catch, and what false alarms would a lab see?

## What we did

Reproduced the real-run side of version 6 of the "error to add" analysis exactly, scored the 20-task pilot of simulated honest runs with the committed scoring code, and computed the error under each setting's own thresholds and false-alarm diagnostics, with size-matched yardsticks made by swapping batches of real honest runs (the third honest batch against the second real honest arm, and the first arm against the second arm and the third batch together).

## What we found

Pilot reading (post hoc, n = 20; RULE.md, pilot reading): two of three error-to-add estimates and three real-honest false-alarm deviations under simulated calibration exceeded their matched real-run reference bounds; preliminary, with only 20 tasks; weak, threshold noise not fully covered at n = 20.

## In the post?

Not in the post.

## Status

Superseded by [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) (the final pooled analysis, per-setting thresholds on all 1,313 tasks)

## Builds on

[2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/) (v6); [2026-10-14_q3_q35_simhonest_purchase](../2026-10-14_q3_q35_simhonest_purchase/)

## Main scripts

- [`run_simhonest_a25.py`](run_simhonest_a25.py)

Result files the index names:

- `a25_pilot_2026-10-14_204219.json` (not copied: result file 1 MB or larger)
- `a25_reference_A_vs_BuC_2026-10-14_204336.json` (not copied: result file 1 MB or larger)
- `a25_reference_C_vs_B_2026-10-14_204319.json` (not copied: strings over 300 characters)
- `a25_pipeline_test_2026-10-14_171757.json` (not copied: result file 1 MB or larger)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_simhonest_a25_09/` at commit `e2b4a538b976e1e0a73f5b96536e735dc4af3d6f` of our internal repository. 28 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
