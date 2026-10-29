# 2026-10-13_pooled_world_removal_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

If a detector is trained on pooled real and simulated honest runs and the honest real-vs-simulated directions are projected out, does the simulated-minus-real catch gap close (is the giveaway one linear direction)?

## What we did

Qwen3-32B tool-output probes (averaged and last-token readouts, layer 48) reading GPT-5.6-Sol runs: pooled honest training with k = 1 to 10 world directions removed (logistic nullspace or mean difference), compared with full-count and count-matched real-trained detectors; repeated on Seed-OSS-36B. Modal H100, about \$0.66 plus amendments.

## What we found

README (printout q1b_20261013_233102): reading 'if not', the giveaway is not one linear direction; removing 10 directions moves the pooled simulated-minus-real gap by -2.9 points on the averaged readout (from +13.3) and -0.9 on the last token (from +17.8) at 1%; mean-difference removal forces held-out realism to chance by construction (correction appended). README caveat: removing 10 directions does not bring realism to chance, and the gap that persists at k = 10 (+10.4 averaged, +17.0 last token, at 1%) cannot be read as a strong "not that direction".

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (another working session's full-fix packs); [2026-10-13_world_direction_POSTHOC](../2026-10-13_world_direction_POSTHOC/); [2026-10-13_matched_world_null_POSTHOC](../2026-10-13_matched_world_null_POSTHOC/)

## Main scripts

- [`run_q1b.py`](run_q1b.py)
- [`print_q1b.py`](print_q1b.py)
- [`print_q1b_fulla.py`](print_q1b_fulla.py)

Result files the index names:

- [`q1b_20261013_233102.printout.txt`](results/q1b_20261013_233102.printout.txt)
- [`q1bfa_20261013_234256.printout.txt`](results/q1bfa_20261013_234256.printout.txt)
- [`q1bseed_20261014_000540.printout.txt`](results/q1bseed_20261014_000540.printout.txt)

## Files here

3 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-13_pooled_world_removal_POSTHOC/` at commit `da99ac0d58e4546cfc1e1032aed25c38d4a82e38` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
