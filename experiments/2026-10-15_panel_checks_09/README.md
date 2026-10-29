# 2026-10-15_panel_checks_09

2026-10-15 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Two checks raised by a review panel of seven language models: how often does the "error to add" estimator, which floors negative terms at zero, report zero when the true error is about 5 points, and do results differ between probes with high and low catch?

## What we did

Counted, in the stored datasets of the earlier coverage simulation, how often the floored estimate was exactly zero, and reran the version-6 pilot analysis split by the probes' catch level; an independent recheck followed.

## What we found

True error in (4, 6] points: estimator read 0 in 10% (tool outputs 1%), 5% (tool outputs 5%) and 26% (commands 5%) of datasets (recheck/RECHECK.md, 3 of 3 MATCH).

## In the post?

Not in the post.

## Status

Exploratory (estimator not in the post)

## Builds on

[2026-10-15_u_coverage_09](../2026-10-15_u_coverage_09/)

## Main scripts

- [`count_false_zero.py`](count_false_zero.py)
- [`run_xerr_catchsplit.py`](run_xerr_catchsplit.py)
- [`table_catchsplit.py`](table_catchsplit.py)

Result files the index names:

- `false_zero_2026-10-15.json` (not copied: result left out to keep the release under 120 MB)
- `catchsplit_v6_2026-10-15_010706.json` (not copied: result file 1 MB or larger)

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_panel_checks_09/` at commit `65510bf270d90eefd14ec37642c7fdb165e565f2` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
