# 2026-10-15_cut_context_quote

2026-10-15 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

How much would it cost to retrain and rescore without the 32,768-token context limit?

## What we did

Counted epochs per root, tokens before the cut and long-epoch H200 extraction timings from staging and extraction records (keys and counts only) to quote a full-length rerun.

## What we found

A cost quote (counts only); fed the decision to re-extract at full length.

## In the post?

Not in the post.

## Status

Current (one-off quote)

## Builds on

[2026-10-14_q3_overnight_chain](../2026-10-14_q3_overnight_chain/) extraction records

## Main scripts

- [`make_cut_quote.py`](make_cut_quote.py)

Result files the index names:

- `cut_context_quote_2026-10-15.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_cut_context_quote/` at commit `f8483fcc53d4b3d56c89a61c1c47e0c15dd5a694` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
