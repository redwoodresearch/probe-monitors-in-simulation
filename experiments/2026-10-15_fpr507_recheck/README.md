# 2026-10-15_fpr507_recheck

2026-10-15 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does an independent checker reproduce the 507-task unit-level false-alarm sweep?

## What we did

Recomputed every row of the registered 16-row sweep from its spec and pinned inputs in a Modal CPU job; compared every field by key; five runs as readings of spec-silent points were identified.

## What we found

Run 5: MATCH, 16 of 16 rows, 0 mismatches over about 209,000 compared leaves (README).

## In the post?

Not in the post (checks an earlier sweep before the final pooled analysis).

## Status

Superseded target (the 507 sweep was replaced by the final pooled records)

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) (registered507_2026-10-15_103452)

## Main scripts

- [`run_fpr507_recheck.py`](run_fpr507_recheck.py)
- [`analyze_cohort_exclusion_keys.py`](analyze_cohort_exclusion_keys.py)

Result files the index names:

- `recheck_2026-10-15_121514.json` (not copied: result left out to keep the release under 120 MB)
- `cohort_exclusion_keys_2026-10-15_124553.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_fpr507_recheck/` at commit `b40cb70d3bec8fd562aa8638af258ff3dbc0cc95` of our internal repository. 26 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
