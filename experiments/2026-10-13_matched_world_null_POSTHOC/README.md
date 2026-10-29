# 2026-10-13_matched_world_null_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

After removing 10 'world' (simulated-vs-real) directions, is the realism signal that remains above a label-shuffle null whose directions are also refit on the shuffled labels?

## What we did

For four settings (world-direction removal at layers 48 and 63; pooled-honest removal on averaged and last-token readouts; Qwen3-32B reading GPT-5.6-Sol honest runs), refit the 10 nullspace directions and the held-out realism probe on each of 20 within-task label shuffles, gated on reproducing the as-run value. Modal H100, \$5.35.

## What we found

README on a side branch (f300b91cd): every job stays 'still above the matched null' (layer 48: 0.5825 vs 95th 0.5219; layer 63: 0.5654 vs 0.5163; averaged primary 0.5394 vs 0.5236, margin only +0.0043); the matched null sits below the unmatched one, so earlier nulls were conservative; p = 1/21 is the floor at 20 shuffles.

## In the post?

Not in the post.

## Status

Exploratory (results only on a side branch and an archive tag)

## Builds on

[2026-10-13_world_direction_POSTHOC](../2026-10-13_world_direction_POSTHOC/); [2026-10-13_pooled_world_removal_POSTHOC](../2026-10-13_pooled_world_removal_POSTHOC/)

## Main scripts

- [`run_matched_null.py`](run_matched_null.py)
- [`fill_matched_null.py`](fill_matched_null.py)

Result files the index names:

- `matched_null_wd_POSTHOC_2026-10-14_003258.printout.txt` (not in this folder at the source commit)
- `matched_null_q1b_POSTHOC_2026-10-14_003258.printout.txt` (not in this folder at the source commit)
- `manifest_matched_world_null_v1.json` (not in this folder at the source commit)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_matched_world_null_POSTHOC/` at commit `cc3a3722f8e2630ffbf605883f9661c25c77d820` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
