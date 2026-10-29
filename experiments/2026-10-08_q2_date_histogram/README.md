# 2026-10-08_q2_date_histogram

2026-10-08 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

How do dates and hours in tool outputs differ between real and simulated runs (a date tell inventory)?

## What we did

Modal CPU count of dates in fully visible tool outputs of the v4 GPT-5.6-Sol inputs (within the 32,768-token cut), per arm, with a seeded sample, paired-overlap and tie analyses; predictions committed first.

## What we found

Simulated attacks show 2025 in 12.1% of outputs vs 3.8% and 4.4% for real runs, and share real runs' most common date far less than two real runs do (-41.2% [-46.8%, -35.5%]); honest depends on batch (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-06_all_layers_extraction](../2026-10-06_all_layers_extraction/) (v4 inputs)

## Main scripts

- [`run_date_histogram.py`](run_date_histogram.py)
- [`summarize_dates.py`](summarize_dates.py)
- [`analyze_paired_overlap.py`](analyze_paired_overlap.py)
- [`analyze_ties.py`](analyze_ties.py)

Result files the index names:

- `manifest_v2.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_v3.json` (not copied: result left out to keep the release under 120 MB)
- `dates_2026-10-08_162309.json` (not in this folder at the source commit)

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-08_q2_date_histogram/` at commit `e021973909878001c860a982422bc6d554916ba1` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
