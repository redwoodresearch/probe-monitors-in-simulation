# 2026-10-15_continuous_score_09

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Would mapping score shifts to catch shifts give tighter error bars on simulation's error than the binary catch gap?

## What we did

Pre-registered and computed a version of the "error to add" (how many points a lab should add to its error bars when it measures probe catch in simulation) that maps continuous score shifts to catch through a kernel density estimate with Silverman's bandwidth, beside the binary version, on the version-6 headline rows of the 196-task pilot, plus variants of the on-policy probe trained on fractions of the tasks; an independent recheck used its own script.

## What we found

Recheck: with-strategy interval width ratios (mapped/binary) 0.62 / 0.54 / 0.72; off-policy probe tool outputs 5%: score shift -0.154 SD [-0.427, +0.107], binary -5.9 [-12.9, +0.65]; training-fraction comparison confounded by a pool change (recheck_002602/RECHECK.md, summary by the analysis owner).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

2026-10-14 error-to-add records, version 6 (outside range)

## Main scripts

- [`run_xerr_cont.py`](run_xerr_cont.py)
- [`run_v7_cont.py`](run_v7_cont.py)
- [`summarize_cont.py`](summarize_cont.py)
- [`contcore.py`](contcore.py)

Result files the index names:

- `xerr_cont_v6_2026-10-15_002602.json` (not copied: strings over 300 characters)
- `v7_cont_2026-10-15_002602.json` (not copied: strings over 300 characters)
- `xerr_cont_v6_2026-10-15_120144.json` (not copied: strings over 300 characters)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_continuous_score_09/` at commit `fbbf6f3596628e3d65fc7c803fb0f00ea86386a4` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
