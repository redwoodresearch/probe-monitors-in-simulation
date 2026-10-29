# 2026-10-14_p2_longcheck

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do the high-stakes and cyber probe training reads reproduce on their longest inputs on a fresh H200 (after B200 numerics were found to break on long contexts)?

## What we did

In one fresh H200 container, re-read the 8 longest items per shard of each stage-1 job with the same capture code and compared all 64 layers by cosine and maximum difference against stored rows (v2 rerun for the cyber one-shard stamp).

## What we found

High-stakes: every longest item bit-identical (cosine 1.000000). Cyber: failed in both runs (6 of 8 items below the 0.9999 cosine bar in the late v2 run), which led to the cyber probe's withdrawal (longcheck printouts; RULE.md).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_p2_highstakes](../2026-10-14_p2_highstakes/); [2026-10-14_p2_cyber](../2026-10-14_p2_cyber/)

## Main scripts

- [`run_longcheck.py`](run_longcheck.py)
- [`run_longcheck_v2.py`](run_longcheck_v2.py)

Result files the index names:

- [`longcheck_2026-10-14_135611.json`](results/longcheck_2026-10-14_135611.json)
- [`longcheck_2026-10-14_154118.json`](results/longcheck_2026-10-14_154118.json)

## Files here

2 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-14_p2_longcheck/` at commit `d2d1e0e76810656b3a5367825e3ec8a3b9eede43` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
