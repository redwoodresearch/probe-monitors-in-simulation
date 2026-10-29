# 2026-10-13_claimB_tell_examples

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

What do three known simulator giveaways look like (insert reply rendered as a diff, Go replies with literal tabs, a factual slip), shown as seeded random examples rather than hand-picked ones?

## What we did

Drew one example of each giveaway at random with a stated seed from honest runs only (fully fixed simulated honest vs real honest on the same task) on Modal CPU, for our internal status page and S3; no raw text in git. \$0.001.

## What we found

Only a manifest and a post-launch review are committed (tell_examples_manifest_2026-10-13.json); the examples themselves are on S3 and our internal status page.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_claimB_tab_census](../2026-10-13_claimB_tab_census/); [2026-10-13_claimB_read10](../2026-10-13_claimB_read10/); [2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/)

## Main scripts

- [`run_tell_examples_modal.py`](run_tell_examples_modal.py)

Result files the index names:

- `tell_examples_manifest_2026-10-13.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `codex_postlaunch_review_2026-10-13_2256.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_claimB_tell_examples/` at commit `f22f3b41ad39a4e1645729257c4e7498dcf6dab8` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
