# 2026-10-14_q3_scaleup_canary

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Before paying for the 311-task scale-up, does the pipeline for Qwen3.5-27B's own runs work on tasks outside the 196 pilot tasks, and at what completion rate, side-task attempt rate and cost?

## What we did

Launched a 20-task trial (2 per environment, drawn by hash from the list of the paper's 804 other tasks) with the frozen September launch code on a third remote development machine: per task a first and then a second real honest run, two real attack runs and one simulated attack run, with preflight, settings-equality and hash checks and a \$650 hard spending stop.

## What we found

Gate 1 preflight 40 of 40 and 390 of 392 (per the experiment's written rules); the scale-up's written rules say the 8 canary cells in the list of +311 added tasks are not re-run and their canary runs count. Completion and attempt counts are not committed in this folder.

## In the post?

In the post: Appendix A runs (its 8 cells inside the list of +311 added tasks count as runs of the +311 added tasks)

## Status

Current

## Builds on

[2026-10-14_q2_scaleup_cohort](../2026-10-14_q2_scaleup_cohort/)

## Main scripts

- [`select_canary.py`](select_canary.py)
- [`canary.sh`](canary.sh)
- [`prelaunch_gates.sh`](prelaunch_gates.sh)
- [`gate_settings.py`](gate_settings.py)
- [`negative_tests.sh`](negative_tests.sh)
- [`check_cross_arm.py`](check_cross_arm.py)

Result files the index names:

- `preflight_2026-10-14_131200.json` (not copied: result left out to keep the release under 120 MB)
- [`negative_tests_2026-10-14_1359.txt`](results/negative_tests_2026-10-14_1359.txt)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

18 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_q3_scaleup_canary/` at commit `15a171d1241fae3b7582fb3fdeb53ede9f0b040e` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
