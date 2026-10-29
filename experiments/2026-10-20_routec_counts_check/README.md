# 2026-10-20_routec_counts_check

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do the count and share numbers in the final honest-detector and simulated-pool records reproduce independently?

## What we did

A fresh checker re-derived counts from pinned inputs under a pre-stated rule (never reading the producers' code); shares compared only inside a Modal container that returned booleans.

## What we found

317 of 321 items equal; all 192 independent items equal; the 4 unequal items are labelled definition differences (result summary).

## In the post?

Not in the post (verifies Table 7 rows 4-5 cohorts and the footnotes on simulator fallback and failed runs).

## Status

Current

## Builds on

[2026-10-16_honest_sim_detector_all](../2026-10-16_honest_sim_detector_all/); [2026-10-20_routec_simpool_numbers](../2026-10-20_routec_simpool_numbers/)

## Main scripts

- [`countscheck.py`](countscheck.py)
- [`run_routec_counts_check.py`](run_routec_counts_check.py)
- [`launch_countscheck.zsh`](launch_countscheck.zsh)

Result files the index names:

- `routec_counts_check_2026-10-20_033953.json` (not copied: strings over 300 characters)
- `pilot_header_bits_2026-10-20_033953.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_counts_check/` at commit `41129b974c113a5449323986e2e76be000864527` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
