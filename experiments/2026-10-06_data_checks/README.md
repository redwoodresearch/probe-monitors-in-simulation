# 2026-10-06_data_checks

2026-10-06 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

What was the simulator shown as its reference transcript, which run versions should be used, can single steps be simulated from a real prefix, and do per-action attack labels exist for the 2026-09-18 LinuxArena cohorts on ten public environments?

## What we did

Read-only checks (no spend) of the real and simulated GPT-5.6-Sol LinuxArena cohorts: where the simulator's reference transcripts came from (200 real honest runs that are also in the compared real cohort, or 800 fresh honest rollouts), original versus repaired run versions (repaired sets redrew some empty or unscored runs), support for simulating a single step, per-action labels, and an existing pairwise AI-judge baseline; with answers from a collaborator and decisions from a meeting with the project manager.

## What we found

Of 939 real/simulated pairs, 196 simulated runs used references identical to real honest runs compared against and 743 used fresh ones; the existing GPT-5.6-Sol pairwise judge told real from simulated at 0.62 [0.59, 0.65]. Decision: stay on v1 runs; no per-action attack labels exist.

## In the post?

Not in the post.

## Status

Current

## Builds on

Nothing earlier in this index.

## Main scripts

- none (README only)

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-06_data_checks/` at commit `016887de7832a26fd6b9f3412ec75985bae82d1c` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
