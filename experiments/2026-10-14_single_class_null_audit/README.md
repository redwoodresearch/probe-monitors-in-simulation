# 2026-10-14_single_class_null_audit

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Which claims on the project's claim table and internal status page that read "within its null" rest on a within-task label shuffle that cannot move labels in single-class tasks (tasks with only attack or only honest runs), so that the null is inflated?

## What we did

Searched our internal status page for fixed phrases, traced each hit to its script, and ran one Modal CPU job counting the single-class units in each analysis, gated on reproducing an earlier count (542 tasks, 105 attack-only, 6 honest-only).

## What we found

The claim table has none; on our internal status page 8 claims are possibly anti-conservative (6 feeding the write-up); about 30 other readings are safe (README).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-13_whitening_POSTHOC](../2026-10-13_whitening_POSTHOC/)

## Main scripts

- [`count_single_class.py`](count_single_class.py)
- [`run_count_modal.py`](run_count_modal.py)

Result files the index names:

- `counts_2026-10-14_010429.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_single_class_null_audit/` at commit `f7683ef9e2b8d25455c6519d23b78a106164c298` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
