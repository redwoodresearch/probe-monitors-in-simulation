# 2026-10-16_own_all_rs_simtrained

2026-10-16 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Pin the score files for an on-policy probe trained on simulated runs (simulated attack vs simulated honest).

## What we did

Keys-only pins mapping each full-length on-policy score folder to its simulated-trained counterpart (Modal CPU), with Codex reviews.

## What we found

First launch crashed at container import; a fix was committed (spend row); no result file in this folder.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-16_own_all_rs](../2026-10-16_own_all_rs/)

## Main scripts

- [`run_simt_pins.py`](run_simt_pins.py)

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_own_all_rs_simtrained/` at commit `460acc84dac4fcd256fa222b6ff632532e18bdc1` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 1 main script that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: low). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
