# 2026-10-15_q3_interleaved_run3_sim2

2026-10-15 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Do simulated and real attack runs track when they run in the same launch? (Also holds the launch scripts for the remaining 493 paper tasks' simulated attacks.)

## What we did

Launched one third real attack and one second simulated attack per cell in a single interleaved launch on the original task set, the +311 added tasks and the new-main-task set (approved ceiling about \$24.5k); the folder also carries the simulated-attack follow-up launchers for the paper's remaining 493 tasks and their reference pins.

## What we found

Launch artifacts (configs, watchers, run tables); activation dumps logged in ACTIVATIONS.md (entries for the interleaved launch on the new-main-task set and for the paper's other 493 tasks).

## In the post?

In the post: indirect: the simulated attack arm of the 493 paper tasks in Figure 1 (the interleaved runs themselves are not in the post)

## Status

Current for the simulated arm of the paper's remaining 493 tasks; interleaved arms exploratory

## Builds on

[2026-10-14_il_launch_09](../2026-10-14_il_launch_09/) key lists

## Main scripts

- [`stage_task_files.py`](stage_task_files.py)
- [`il311_box.sh`](il311_box.sh)
- [`il_nm_box.sh`](il_nm_box.sh)
- [`sp493sim_box.sh`](sp493sim_box.sh)
- [`build_sp493_tail_records.py`](build_sp493_tail_records.py)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

214 code and configuration files and 10 result files, copied byte for byte from `experiments/2026-10-15_q3_interleaved_run3_sim2/` at commit `3157f33122ca68b15a612599ae68be41580b272e` of our internal repository. 314 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
