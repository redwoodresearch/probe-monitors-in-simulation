# 2026-10-14_launch_effect_joint_09

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Is there a systematic difference between real run 2 and real run 1 (a launch effect), tested per probe and jointly across probes in a way that respects their shared runs?

## What we did

Per-task sign-flip permutation tests (100,000 draws, same flip for every probe) of run 2 minus run 1 on continuous scores and on catch at 1% and 5%, for five probe-readouts (harmfulness, refusal, FIBS deception, Sol-trained reply and command) on Qwen3.5-27B's pilot runs, plus joint direction and maximum statistics.

## What we found

No common score shift detected (joint direction test p 0.148, maximum test p 0.190); on catch at 5% the joint direction was −3.8 points (p 0.023) (launch_joint_2026-10-14_133122.json, result.joint).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_p2_refusal_harm](../2026-10-14_p2_refusal_harm/); [2026-10-14_p2_fibs_pair](../2026-10-14_p2_fibs_pair/); Sol-trained sweep record (on a side branch)

## Main scripts

- [`run_launch_joint.py`](run_launch_joint.py)

Result files the index names:

- `launch_joint_2026-10-14_133054.json` (not copied: result left out to keep the release under 120 MB)
- `launch_joint_2026-10-14_133122.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_launch_effect_joint_09/` at commit `39aaeb4bd61f38fd982bf3fbaf95935b9a61392c` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
