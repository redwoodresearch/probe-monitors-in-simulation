# 2026-10-14_priority1_firstcut

2026-10-14 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

A first quick answer to the supervisor's top-priority question: on Qwen3.5-27B's own runs, how tight can a paired comparison of simulated and real catch be, beside the same comparison between its two real runs?

## What we did

An independent working session computed, from stored per-run scores only (the on-policy probe; the off-policy probe trained on GPT-5.6-Sol's runs, averaged over tool outputs and at commands; probes trained on format-fixed simulated runs; off-policy probes adapted with some of Qwen3.5-27B's own runs), catch for each group of runs on the 183-task pilot with one threshold from two pooled batches of real honest runs (A and B), plus paired t and Tango intervals, McNemar tests and differences in standardized scores.

## What we found

Sol-trained averaged probe at 1% (177 tasks): sim − run 2 −2.26 points (95% −9.14 to +4.62) vs run 2 − run 1 +0.56 (−6.94 to +8.07); on-policy probe sim − run 2 +0.00 (−2.74 to +2.74) (firstcut_2026-10-14_120513.printout.txt).

## In the post?

Not in the post.

## Status

Superseded by the priority-1 spec analyses ([2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/)) and finally [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) (the final pooled analysis)

## Builds on

[2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/); [2026-10-14_headline_recheck_09](../2026-10-14_headline_recheck_09/)

## Main scripts

- [`run_firstcut.py`](run_firstcut.py)

Result files the index names:

- `firstcut_2026-10-14_115617.json` (not copied: result left out to keep the release under 120 MB)
- `firstcut_2026-10-14_115701.json` (not copied: result left out to keep the release under 120 MB)
- `firstcut_2026-10-14_120513.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_priority1_firstcut/` at commit `f71112c8b8c02fdf09063616b973b78cb5ccf74b` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
