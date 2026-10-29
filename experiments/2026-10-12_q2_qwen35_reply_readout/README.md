# 2026-10-12_q2_qwen35_reply_readout

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does the reply-readout gap (the in-domain tool-output attack detector catches simulated attack runs less often than real ones) replicate when Qwen3.5-27B, instead of Qwen3-32B, reads the same GPT-5.6-Sol transcripts?

## What we did

Refit the detector recipe on Qwen3.5-27B activations of the 939 paired task cells (whole transcripts, layer chosen per fold) and compared catch on simulated runs with a second real attack run at 5% and 1% false alarms, with a permutation null that swaps simulated and real runs within tasks and planted shifts of plus or minus 10 points to check sensitivity.

## What we found

README: "inconclusive": simulated minus second real -2.7 points [-5.4, +0.1] at 5% (-3.1 [-6.1, -0.5] at 1%), vs -14.3 for Qwen3-32B.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/)

## Main scripts

- [`run_q3_attack_qwen35.py`](run_q3_attack_qwen35.py)
- [`analyze_reply_break.py`](analyze_reply_break.py)

Result files the index names:

- `reply_break_2026-10-12_012917.json` (not in this folder at the source commit)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q2_qwen35_reply_readout/` at commit `1f7d593d03436c5a98af15fa183bf3a87bc4bb3b` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
