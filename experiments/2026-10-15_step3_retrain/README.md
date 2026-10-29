# 2026-10-15_step3_retrain

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Does training the on-policy probe on a mix of both real launches (instead of run 1 only) change the simulated-vs-real gap, i.e. is there a training-launch artifact?

## What we did

Refit the on-policy probe (layer 48) on runs 1+2 vs run 1 only with the same folds and negatives; paired contrast K (the simulated-minus-real catch gap after training on both real launches minus that gap after training on the first launch only) on 90 pilot estimation tasks; then scored these recipes on simulated honest batches and full-length roots (many per-batch addenda).

## What we found

K (the simulated-minus-real catch gap after training on both real launches minus that gap after training on the first launch only) = +3.89 points (95% -4.44 to +11.83), "inconclusive at this n" (analysis_2026-10-15_114727.json).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_p2_family_expansion](../2026-10-14_p2_family_expansion/)

## Main scripts

- [`run_step3.py`](run_step3.py)
- [`analyze_step3.py`](analyze_step3.py)
- [`analyze_step3_snew.py`](analyze_step3_snew.py)
- [`build_cutfull_message.py`](build_cutfull_message.py)

Result files the index names:

- [`analysis_2026-10-15_114727.json`](results/analysis_2026-10-15_114727.json)
- `analysis_snew_2026-10-15_115038.json` (not copied: result left out to keep the release under 120 MB)
- `simh_step3_2026-10-15_120335.json` (not copied: result left out to keep the release under 120 MB)

## Files here

26 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-15_step3_retrain/` at commit `9e7351bbc8e2c80e0cae50df5a4c7648aa43f775` of our internal repository. 783 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
