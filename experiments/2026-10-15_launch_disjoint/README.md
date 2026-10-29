# 2026-10-15_launch_disjoint

2026-10-15 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Which tasks can an on-policy probe score without sharing a launch with its training runs, and how did the attempted-only cohort shrink?

## What we did

Built launch-disjoint eligibility files (v1-v5) for on-policy probe variants and a cohort-attrition table by exclusion class, from keys and labels only.

## What we found

Attempted-only cohort: +311 added tasks 311 to 272, new-main-task set 360 to 312 (244 without the fourth new-main-task launch split) (cohort_attrition md).

## In the post?

Not in the post.

## Status

Superseded by the final 1,313-task cohort (exclusions in [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/))

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) membership710

## Main scripts

- [`make_launch_disjoint_table.py`](make_launch_disjoint_table.py)
- [`make_cohort_attrition.py`](make_cohort_attrition.py)

Result files the index names:

- `launch_disjoint_eligibility_v5_2026-10-15_224056.json` (not copied: result file 1 MB or larger)
- `cohort_attrition_2026-10-15_223144.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_launch_disjoint/` at commit `90c0c31485d5400094683e957e8c7f24136ff76a` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
