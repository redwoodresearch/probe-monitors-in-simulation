# 2026-10-14_p2_dt_scaleup

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Do the call-level monitors for destructive and transfer commands catch simulated attacks as often as real ones on the full scale-up cohort, scored in the reader layout?

## What we did

Re-scored the 196 pilot tasks in the reader layout with the frozen monitor weights, as a bridge test against the earlier served-token layout scores, then scored every scale-up set (the 311 added tasks, 360 new main tasks, interleaved third real and second simulated attack runs, simulated honest runs, and the paper's other 493 tasks) on Modal CPU, with many record, coverage and fold checks and dated launcher snapshots.

## What we found

The bridge test on the 196-task pilot failed its pre-registered bar (Spearman 0.89 and 0.86 against a predicted 0.995; match rate 0.504), so the scale-up numbers of the destructive-action and data-transfer monitors are labelled a new readout (the rules of the call-level error-to-add analysis on 507 cells, quoting dt_q4b_t196 record). Destructive-action and data-transfer monitors are not among the post's six probes.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_p2_callfacet](../2026-10-14_p2_callfacet/); [2026-10-14_q3_overnight_chain](../2026-10-14_q3_overnight_chain/); [2026-10-14_q3_scaleup_311](../2026-10-14_q3_scaleup_311/); [2026-10-14_q3_new_mains_launch](../2026-10-14_q3_new_mains_launch/)

## Main scripts

- [`run_dt_q4b.py`](run_dt_q4b.py)
- [`dt_q4b_lib.py`](dt_q4b_lib.py)
- [`check_dt_unseen.py`](check_dt_unseen.py)
- [`check_dt_training_crossfit.py`](check_dt_training_crossfit.py)
- [`check_sp493_records.py`](check_sp493_records.py)
- [`write_sp493_cells.py`](write_sp493_cells.py)

Result files the index names:

- `dt_training_crossfit_check_2026-10-15_194337.json` (not copied: result left out to keep the release under 120 MB)

## Files here

48 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_dt_scaleup/` at commit `7fe9227b070036c299cabc3fb48278b1073da863` of our internal repository. 410 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
