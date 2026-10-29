# 2026-10-15_e2e_sim_pipeline

2026-10-15 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do the pooled records that carry the main numbers reproduce when re-derived independently from their own inputs?

## What we did

Began as an end-to-end fully simulated pipeline on 183 tasks (each setting with its own thresholds); it then became the home of independent rechecks of the main pooled records' point estimates (before resampling), of checks that re-derive the final records' values with each run scored by its own cross-validation copy of the probe (main results at 1% and 5% alarm levels, the sweep of alarm levels, and strict alarm levels), and of a ported check with planted rows.

## What we found

The check that each run uses its assigned cross-validation model, on the final main pooled results record: 124 rule keys, 0 violating; an earlier coverage note found 32 of 160 public-probe rows had not been compared (NOTE_g7b_routec_2026-10-20.md; coverage NOTE).

## In the post?

Not in the post (verifies the records behind Figure 1, Tables 5/8 and Appendix C).

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) records

## Main scripts

- [`check_stagea_alltasks.py`](check_stagea_alltasks.py)
- [`run_e2e.py`](run_e2e.py)
- [`plant_rows_v3.py`](plant_rows_v3.py)
- [`run_g7b_foldmodel.sh`](run_g7b_foldmodel.sh)

## Files here

41 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-15_e2e_sim_pipeline/` at commit `e4c53f97117a47498ff23ad96bbe196dd479e2a6` of our internal repository. 553 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
