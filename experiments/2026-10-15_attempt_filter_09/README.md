# 2026-10-15_attempt_filter_09

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Does the earlier error-size headline (an estimate of how far simulated catch can sit from real catch) move if tasks are not filtered to those where the agent clearly attempted the side task in all three attack runs?

## What we did

On the original 196-task pilot cohort (Qwen3.5-27B agent, earlier probe set incl. the off-policy probe), reran the frozen v6 headline with two variants: runs labelled "unclear" counted as not caught, and all runs scored as is (190 tasks instead of 183); one Modal CPU job.

## What we found

Base reproduced v6 exactly; off-policy probe catch at 5% (tool outputs, with-strategy tasks) real 95.29/94.32/95.45 and sim 89.41/87.50/88.64 for the original attempted-only cohort / unclear runs counted as not caught / all runs scored as is, respectively (recheck_001612/RECHECK.md, a fresh-agent recheck relayed as a summary). The post later counts all scored runs regardless of attempt.

## In the post?

Not in the post.

## Status

Exploratory (applies to the pilot-era error-to-add headline, which the post dropped)

## Builds on

2026-10-14 error-to-add records, version 6 (outside range)

## Main scripts

- [`run_xerr_attempt.py`](run_xerr_attempt.py)
- [`summarize_attempt.py`](summarize_attempt.py)

Result files the index names:

- `xerr_attempt_sens_2026-10-15_001612.json` (not copied: result file 1 MB or larger)
- [`summary_attempt_sens_2026-10-15_001612.txt`](results/summary_attempt_sens_2026-10-15_001612.txt)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-15_attempt_filter_09/` at commit `3e7f8d790d77788e373b7dec37d33fdb7b0d8a70` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
