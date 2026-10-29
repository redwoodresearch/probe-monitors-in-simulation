# 2026-10-14_p2_served_recheck

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does an independent checker reproduce the served-context scoring (scores, units, counts and hashes) without reading the scorer's code?

## What we did

An independent checker recomputed the second real honest batch's run scores and per-row scores on all three served-context layouts with its own numerical path (numpy float32 and a float64 reference), checked unit sets and assignments on 60 seeded runs, and in an addendum counted how many alarm decisions flip at the thresholds.

## What we found

Run level: 17 of 20 cells within 1e-6 (3 high-stakes cells up to 3.0e-6); row level within float32 accumulation (max 3.3e-5); 0 decision flips at the thresholds in the probes checked (ADDENDUM.md label and section 4b).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_p2_replay_check](../2026-10-14_p2_replay_check/)

## Main scripts

- [`run_recheck.py`](run_recheck.py)
- [`analyze_recheck.py`](analyze_recheck.py)
- [`run_addendum.py`](run_addendum.py)
- [`analyze_addendum.py`](analyze_addendum.py)
- [`run_diag_userturn.py`](run_diag_userturn.py)
- [`run_think_extra.py`](run_think_extra.py)
- [`run_think_keys.py`](run_think_keys.py)
- [`run_schema_probe.py`](run_schema_probe.py)

Result files the index names:

- `recheck_2026-10-14_230545.json` (not copied: result left out to keep the release under 120 MB)
- `addendum_2026-10-15_001609.json` (not copied: result left out to keep the release under 120 MB)
- `think_extra_2026-10-14_233900.json` (not copied: strings over 300 characters)
- `think_keys_2026-10-14_234546.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_served_recheck/` at commit `eb247348dfb6fb6cd26033ef9796270de48696c4` of our internal repository. 20 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
