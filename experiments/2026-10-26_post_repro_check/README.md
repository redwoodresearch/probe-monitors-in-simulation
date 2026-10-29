# 2026-10-26_post_repro_check

2026-10-26 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does the post's one-command rebuild work from a fresh clone?

## What we did

Ran post/rebuild_post.py in a clean Modal container holding only post/ from a fresh shallow clone.

## What we found

Fresh-clone runs 3 and 4 passed (rc 0): 445 of 445 rebuilt numbers and every figure match; runs 1 and 2 had failed (a missing data file, then a one-line rebuild bug), both fixed

## In the post?

Not in the post (reproducibility of the post package).

## Status

Current

## Builds on

[2026-10-25_realline_final_read](../2026-10-25_realline_final_read/); [2026-10-26_public_release](../2026-10-26_public_release/)

## Main scripts

- [`run_fresh_clone_check.py`](run_fresh_clone_check.py)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-26_post_repro_check/` at commit `f60a84bb20e97e50919be7df51ce386ec687e7b9` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
