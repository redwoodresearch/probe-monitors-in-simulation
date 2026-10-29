# 2026-10-14_am22l_recheck

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does the comparison of Qwen3.5-27B's on-policy probe with the probe trained on GPT-5.6-Sol's runs (196-task pilot, corrected transcript rendering) reproduce when an independent agent recomputes it from the raw scores with new code?

## What we did

An independent agent wrote fresh code from the pre-registered rule text and recomputed on Modal CPU the catch rates and differences (on-policy probe vs the GPT-5.6-Sol-trained probe, and vs a count-matched variant) at 1% and 5% false alarms with 2,000 task resamples, then compared key by key with the stored record correctedb_am22l_2026-10-14_182503.json in [2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/).

## What we found

Reproduced: 0 mismatches in 1,017 comparisons, 0 secondary mismatches (am22l_recheck_2026-10-14_183729.json, keys mismatches_total, comparisons_total).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/)

## Main scripts

- [`run_recheck_am22l.py`](run_recheck_am22l.py)

Result files the index names:

- `am22l_recheck_2026-10-14_183729.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_am22l_recheck/` at commit `01c14080bf32a8eb2371b16c1f99293b71b3276d` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
