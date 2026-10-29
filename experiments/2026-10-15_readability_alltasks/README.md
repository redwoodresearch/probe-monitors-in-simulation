# 2026-10-15_readability_alltasks

2026-10-15 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

At which strict alarm levels are thresholds readable on all task sets, and how many more simulated honest runs would make them readable?

## What we did

Counted, per probe and level, whether each honest pool has at least 10 units above threshold (coded flags only) and projected extra original-set simulated honest runs needed.

## What we found

E.g. off-policy probe on commands readable on the original task set and the +311 added tasks from 0.237% upward; at 0.1% the original set's real pools are short (table md).

## In the post?

Not in the post (planning for Appendix C's stricter levels).

## Status

Exploratory

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) sweep records

## Main scripts

- [`count_readability.py`](count_readability.py)

Result files the index names:

- `readability_alltasks_2026-10-15_215745.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_readability_alltasks/` at commit `ea4517635a5d046009d43ffe37d49aade254d236` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
