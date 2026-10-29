# 2026-10-11_q2_transfer_rvr

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

For the data-transfer monitor on GPT-5.6-Sol's runs, how far apart are two real runs per task, and what gap could the data detect?

## What we did

\$0 on the stored separate-calibration refit: real run 1 minus run 2 (163 same-batch pairs, pooled and per task), real minus simulated, and the smallest detectable gap at 80% power.

## What we found

README: real run 1 minus run 2 at 1% +1.8 points [-6.8, +12.4]; real minus simulated +6.8 [-1.0, +12.5] pooled, +9.4 [+0.7, +15.8] counting each task once; smallest detectable real-versus-real gap 13.6 points.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-11_q2_margins_POSTHOC](../2026-10-11_q2_margins_POSTHOC/) (refit of the destructive and transfer monitors)

## Main scripts

- [`analyze_transfer_rvr.py`](analyze_transfer_rvr.py)
- [`analyze_sizing_endpoint_POSTHOC.py`](analyze_sizing_endpoint_POSTHOC.py)

Result files the index names:

- [`transfer_rvr_2026-10-11_133901.json`](results/transfer_rvr_2026-10-11_133901.json)
- [`transfer_rvr_2026-10-11_132002.json`](results/transfer_rvr_2026-10-11_132002.json)
- [`sizing_endpoint_POSTHOC_2026-10-12_040802.json`](results/sizing_endpoint_POSTHOC_2026-10-12_040802.json)

## Files here

2 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_q2_transfer_rvr/` at commit `b86d0f267b6ae2cda528c80943e97bca8d4b7ff0` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
