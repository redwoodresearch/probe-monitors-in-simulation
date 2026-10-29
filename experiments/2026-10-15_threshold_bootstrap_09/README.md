# 2026-10-15_threshold_bootstrap_09

2026-10-15 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

Do error bars that also resample the alarm threshold change the upper limit on simulation's error?

## What we did

Added resampling of the alarm thresholds (jointly with the tasks, and independently) to the bootstrap of the "error to add" (how many points a lab should add to its error bars when it measures probe catch in simulation) on the version-6 headline rows of the pilot tasks.

## What we found

Tool outputs 1% pooled: upper limit 9.78 to 9.87 with joint threshold resampling (summary txt).

## In the post?

Not in the post (the post's intervals re-estimate thresholds per resample).

## Status

Superseded by the joint threshold-resampling bootstrap in [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/)

## Builds on

2026-10-14 error-to-add estimate, version 6

## Main scripts

- [`run_xerr_rt.py`](run_xerr_rt.py)
- [`summarize_xerr_rt.py`](summarize_xerr_rt.py)

Result files the index names:

- `xerr_rt_real_2026-10-15_003201.json` (not copied: result left out to keep the release under 120 MB)

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-15_threshold_bootstrap_09/` at commit `695c7814c4bd7b8979082364f6ea10a1e619fed2` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
