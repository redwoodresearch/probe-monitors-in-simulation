# 2026-10-14_sol_aprime_correctedB_09

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does refitting the off-policy probe (trained on GPT-5.6-Sol's runs) without the 41 training runs that lost earlier reasoning blocks under the old transcript rendering move the headline on correctly rendered runs by 5 points or more?

## What we did

Scored the already-fitted refit's weights on Qwen3.5-27B's correctly rendered pilot runs plus the third batch of real honest runs (C) on Modal CPU, and compared like for like with the stored scores of the original off-policy probe.

## What we found

The refit moves the headline by at most 3.89 points ("Sol-side leak checked"), under the 5-point bar, so no re-staging (compare_aprime_vs_stored_2026-10-14_174736.json, decision).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/)

## Main scripts

- [`run_aprime_score.py`](run_aprime_score.py)
- [`compare_aprime_vs_stored.py`](compare_aprime_vs_stored.py)

Result files the index names:

- `aprime_2026-10-14_174046.json` (not copied: result left out to keep the release under 120 MB)
- `compare_aprime_vs_stored_2026-10-14_174736.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_sol_aprime_correctedB_09/` at commit `6b85dd4710aa1010eee8736717a00a26d1059f03` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
