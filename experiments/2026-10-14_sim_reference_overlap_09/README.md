# 2026-10-14_sim_reference_overlap_09

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

How much of a simulated run's tool-output text is copied from the real honest run the simulator used as a reference?

## What we did

Counts-only Modal CPU job over Qwen3.5-27B's on-policy staging: per run, the share of 8-word sequences and of exact tool outputs that also occur in the same task's run of the first real honest arm, for simulated, real attack and second-arm real honest runs, with a cross-task control.

## What we found

Simulated runs reuse about as much of the reference as real runs: 8-word overlap sim − run 1 +1.6 points (95% −1.4 to +4.6); exact-output reuse 22.6% sim vs 20.2% run 1 and 31.2% for the second real honest arm (overlap_2026-10-14_120033.printout.txt).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_q3_onpolicy_titlecells](../2026-10-14_q3_onpolicy_titlecells/) (staging)

## Main scripts

- [`run_overlap.py`](run_overlap.py)

Result files the index names:

- [`overlap_2026-10-14_120033.json`](results/overlap_2026-10-14_120033.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_sim_reference_overlap_09/` at commit `28cf90024f21a5b5dfbc18123010f057d07da3ea` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
