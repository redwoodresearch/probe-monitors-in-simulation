# 2026-10-11_claimB_review_checks

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Post hoc counts answering pull-request review threads on the fixed-simulator test: are the AI judge's answer pair numbers valid, how much of the per-token attribution's "agent text or template" share comes from the initial prompt, and do the meaning control's four versions keep the same outputs?

## What we did

Free metadata-only scripts on stored judge answers and the per-token attribution's span metadata (which parts of each run the probe score came from), plus one small Modal job on the meaning control's kept-output sets; counts only, no text.

## What we found

Commits: judge answers have 0 repeated and 0 out-of-range pair numbers (a282c9ae4); the meaning control's four versions keep the same outputs in 147 of 150 runs (1df51f6ba); the initial-prompt share result is in review_counts (commit f7badb489).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (describe judge, attribution D); [2026-10-10_claimB_meaning_control](../2026-10-10_claimB_meaning_control/)

## Main scripts

- [`analyze_POSTHOC_judge_pair_numbers.py`](analyze_POSTHOC_judge_pair_numbers.py)
- [`analyze_POSTHOC_review_counts.py`](analyze_POSTHOC_review_counts.py)
- [`run_POSTHOC_mc_output_sets_modal.py`](run_POSTHOC_mc_output_sets_modal.py)

Result files the index names:

- `judge_pair_numbers_POSTHOC_2026-10-11_063527.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `review_counts_POSTHOC_2026-10-11_062205.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `mc_output_sets_POSTHOC_2026-10-11_125010.json` (not in this folder at the source commit)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_review_checks/` at commit `a282c9ae4132e7cae5ecb4ae58424bb09decbc35` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 1 main script that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
