# 2026-10-11_q2_fpr_per_task

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Counting each task once, do real and simulated GPT-5.6-Sol runs get the same false-alarm rate from the impactful-command probe, and where does the quoted '1.1-1.2%' come from?

## What we did

\$0 on stored off-policy scores: per-task and pooled per-call false alarms at the real-trained probe's frozen real 1% threshold, plus a recomputation of the threshold-in-simulation rates.

## What we found

README: per task real 1.03% vs simulated 0.52% (paired difference +0.75 points [-0.09, +1.81]); pooled per call 0.86% vs 0.85%; '1.1-1.2%' is 1.07% and 1.23% from the two off-policy probes, recomputed exactly.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-09_q2_threshold_in_sim](../2026-10-09_q2_threshold_in_sim/); [2026-10-09_q2_tpr_at_1pct](../2026-10-09_q2_tpr_at_1pct/)

## Main scripts

- [`analyze_fpr_per_task.py`](analyze_fpr_per_task.py)

Result files the index names:

- `fpr_per_task_2026-10-11_123146.compact.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_q2_fpr_per_task/` at commit `02fda78181c624ca7a99701a279bb4580ee033e8` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
