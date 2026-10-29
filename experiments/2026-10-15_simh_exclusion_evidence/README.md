# 2026-10-15_simh_exclusion_evidence

2026-10-15 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Why were particular simulated honest (and attack) runs excluded at staging, and does each exclusion have evidence?

## What we did

Built per-cell evidence rows (attempt kinds, exit codes, error class names; no text) from the boxes and committed censuses for excluded simulated runs, including rate-limit reruns.

## What we found

Per-box census and evidence files written; counts not read for this index.

## In the post?

Not in the post (feeds the exclusion classes behind the failed-run footnote).

## Status

Current

## Builds on

[2026-10-15_q3_simhonest_311](../2026-10-15_q3_simhonest_311/)

## Main scripts

- [`simh_excl_evidence.py`](simh_excl_evidence.py)
- [`simh_census.py`](simh_census.py)
- [`reconcile_staging_exclusions.py`](reconcile_staging_exclusions.py)
- [`build_row_attack_excl.py`](build_row_attack_excl.py)

Result files the index names:

- `_2026-10-15.json` (not in this folder at the source commit)

## Files here

12 code and configuration files and 19 result files, copied byte for byte from `experiments/2026-10-15_simh_exclusion_evidence/` at commit `42c0f035404888b92a4d627802c06716bb4128f0` of our internal repository. 103 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
