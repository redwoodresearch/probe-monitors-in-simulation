# 2026-10-14_single_class_rechecks_POSTHOC

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do two on-policy results flagged by the single-class audit survive a label shuffle across all runs, or a within-task shuffle with single-class tasks dropped? The two are the null for the slope of the prefix sweep (probes trained on only the first k tool outputs of each run) and the null for the off-policy probe trained on GPT-5.6-Sol's runs and tested on Qwen3.5-27B's own runs (cell (a)).

## What we did

Re-ran the prefix sweep's label-shuffle null (with a GPU refit on Modal) and the off-policy probe's nulls (from stored scores) with the two alternative shuffles.

## What we found

No status-page reading changed; one secondary is borderline: cell (a) unswapped with times and dates masked, 1% catch beyond the within-task null (p 0.030, also with single-class tasks dropped) but not beyond a global shuffle (p 0.056) (README TL;DR).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_single_class_null_audit](../2026-10-14_single_class_null_audit/)

## Main scripts

- [`run_t1sweep_recheck.py`](run_t1sweep_recheck.py)
- [`run_t1sweep_recheck_gp.py`](run_t1sweep_recheck_gp.py)
- [`run_cella_drop.py`](run_cella_drop.py)
- [`run_cella_global.py`](run_cella_global.py)
- [`print_cella_global.py`](print_cella_global.py)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_single_class_rechecks_POSTHOC/` at commit `56b2bf875a7adcd923d4411b90597f330c85c36f` of our internal repository. 24 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
