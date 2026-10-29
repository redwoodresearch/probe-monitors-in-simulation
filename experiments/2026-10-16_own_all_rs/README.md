# 2026-10-16_own_all_rs

2026-10-16 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Set up the on-policy probe trained on all task sets with a fresh five-fold split by main task, and document its coverage.

## What we did

Built the re-split fold list by main task used by the on-policy probe and the detectors, plus notes on coverage gaps, full-length staging, relaunch members, a row-order shuffle test and a sample-error scan (training code lives in [2026-10-14_p2_family_expansion](../2026-10-14_p2_family_expansion/)).

## What we found

Fold list written; notes only otherwise.

## In the post?

In the post: indirect: the on-policy probe's folds (Table 1, Figure 1) and the detector folds (Figure 2)

## Status

Current

## Builds on

[2026-10-14_p2_family_expansion](../2026-10-14_p2_family_expansion/)

## Main scripts

- [`make_own_all_rs_folds.py`](make_own_all_rs_folds.py)
- [`run_shuffle_refit.py`](run_shuffle_refit.py)
- [`run_uncut_training_mains_union.py`](run_uncut_training_mains_union.py)

Result files the index names:

- `own_all_rs_folds_2026-10-16.json` (not copied: result left out to keep the release under 120 MB)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

18 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_own_all_rs/` at commit `519bc655901f033d04fb2415175b7420a338f812` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
