# 2026-10-14_p2_cyber

2026-10-14 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Does a cyber-offense intent probe (a public stand-in for deployed cyber-misuse probes), fit on CyberSecEval on Qwen3.5-27B, catch simulated attacks as often as real ones?

## What we did

Fit a probe on Meta's CyberSecEval MITRE prompts vs benign prompts from Qwen3.5-27B activations (one H200 shard after an anchor-gate stop), gated on a positive control, then scored the 196-task pilot runs.

## What we found

Withdrawn: the positive control was saturated (64 of 64 layers at AUROC 1.000) and a length-only baseline separated the data (0.794), so the gate measured dataset discrimination; the required longest-input H200 numerics gate, run late, failed (6 of 8 items below 0.9999 cosine). The scores are kept but excluded (RULE.md verdicts).

## In the post?

Not in the post.

## Status

Void (withdrawn)

## Builds on

[2026-10-14_p2_longcheck](../2026-10-14_p2_longcheck/)

## Main scripts

- [`run_read_fit.py`](run_read_fit.py)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_p2_cyber/` at commit `9f4fe1f4023e78fa2d66eb9458926d7e7916d4c5` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
