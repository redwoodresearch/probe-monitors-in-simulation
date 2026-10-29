# 2026-10-16_union_vs_perorigin_check

2026-10-16 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do intervals change if bootstrap draws are taken per task set instead of jointly across all sets?

## What we did

Tabulated points and intervals from the main pooled results record built on runs cut at 32,768 tokens for bootstrap draws taken jointly across all task sets and for draws taken separately within each task set, with jq only.

## What we found

The headline already used bootstrap draws jointly across all task sets, so it does not change; twins with draws taken separately within each task set differ in weighting (README).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) 32k main record

## Main scripts

- [`tab.jq`](tab.jq)

Result files the index names:

- [`tab_2026-10-16_095400.tsv`](results/tab_2026-10-16_095400.tsv)
- [`cohort_check_2026-10-16_095400.tsv`](results/cohort_check_2026-10-16_095400.tsv)

## Files here

1 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-16_union_vs_perorigin_check/` at commit `a62b21a50261dbec8e95a594edcc2f2b8a72d31b` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
