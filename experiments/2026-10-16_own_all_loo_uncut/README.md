# 2026-10-16_own_all_loo_uncut

2026-10-16 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Does the on-policy probe behave the same when the task set it scores is left out of its training (leave-one-origin-out)?

## What we did

Eligibility freezes and coverage checks for leave-one-set-out on-policy probe variants (three and four sets) on full-length activations; a held plan for scoring pool runs of the paper's remaining 493 tasks.

## What we found

Coverage checks against the frozen inputs written; no result values read here.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-16_own_all_rs](../2026-10-16_own_all_rs/); [2026-10-15_own_all_loo](../2026-10-15_own_all_loo/)

## Main scripts

- [`make_loo_uncut_eligibility.py`](make_loo_uncut_eligibility.py)
- [`analyze_loo3_coverage_110447.py`](analyze_loo3_coverage_110447.py)

Result files the index names:

- `loo3_coverage_110447_2026-10-16_165107.json` (not copied: strings over 300 characters)
- `loo_uncut_eligibility_freeze_2026-10-16_170045.json` (not copied: strings over 300 characters)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_own_all_loo_uncut/` at commit `e2003d463894b9688494cfea586e8338922aa3df` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
