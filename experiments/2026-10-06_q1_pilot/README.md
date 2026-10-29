# 2026-10-06_q1_pilot

2026-10-06 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Can a linear probe on Qwen3-32B pooled activations tell real from simulated LinuxArena attack transcripts (first pilot)?

## What we did

Modal CPU analysis script: L2 logistic regression on per-segment mean and moving-average-max pooled activations at 8 layers, main-task folds, real epoch 1 vs epoch 2 null and length, trailing-newline and random-direction baselines.

## What we found

No result exists: the pilot analysis never saved any numbers.

## In the post?

Not in the post.

## Status

Superseded by [2026-10-06_q1_segment_end](../2026-10-06_q1_segment_end/) (segment-end readout)

## Builds on

[2026-10-06_activation_extraction](../2026-10-06_activation_extraction/)

## Main scripts

- [`analyze_q1_pilot.py`](analyze_q1_pilot.py)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-06_q1_pilot/` at commit `680eb76797f927dfa4cbc4d00ca64ee16deb33cb` of our internal repository. Every file in that folder was copied.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
