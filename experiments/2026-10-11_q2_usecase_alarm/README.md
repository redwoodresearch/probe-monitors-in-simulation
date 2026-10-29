# 2026-10-11_q2_usecase_alarm

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

If the 196-task command monitor's 1% alarm is set on simulated runs and deployed on real runs, what real catch and false alarms result, next to an alarm set on real runs?

## What we did

\$0 on stored Qwen3.5-27B on-policy scores: thresholds set on simulated runs, real run 1, real run 2 and the second real honest arm, each applied to real read-only and impactful calls, with task-resampled intervals and planted checks.

## What we found

README: simulated-set alarm gives real false alarms 1.75% [1.32, 3.04] and real catch 83.8% [78.8, 95.3], vs 2.62% and 94.8% for an alarm set on real run 2; catch cost -11.1 points [-15.8, +2.4]; band reading inconclusive; not blind (the 1.75% was public before the plan).

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/); [2026-10-09_q2_threshold_in_sim](../2026-10-09_q2_threshold_in_sim/)

## Main scripts

- [`analyze_usecase.py`](analyze_usecase.py)

Result files the index names:

- `usecase_2026-10-11_195624.json` (not in this folder at the source commit)
- `s3_manifest_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_q2_usecase_alarm/` at commit `601f652a69d559e4abd534b9b5c59034018ed08d` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
