# 2026-10-21_opt3_results

2026-10-21 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Results of the clean-room bootstrap of the final interval ends (the independent bootstrap reproduction of interval endpoints) and its recheck of the alarm-level sweep record.

## What we did

Run logs and summaries of the bootstrap engine for the independent bootstrap reproduction of interval endpoints, on the records of the main pooled results at 1% and 5% alarm levels and of the alarm-level sweep, plus a separate recheck (b) of the alarm-level sweep record in two versions.

## What we found

Intervals of the main pooled results at 1% and 5% alarm levels reproduced exactly and the alarm-level sweep record ruled by the analysis owner (relayed in [2026-10-16_cleanroom_pooled_repro](../2026-10-16_cleanroom_pooled_repro/) opt3/FOLLOWUPS_OPT3.md); summaries here hold launch metadata only.

## In the post?

Not in the post (verifies interval ends for Figure 1 / Tables 5 and 8).

## Status

Current

## Builds on

[2026-10-16_cleanroom_pooled_repro](../2026-10-16_cleanroom_pooled_repro/)

## Main scripts

- [`recheck_b_v3.py`](recheck_b/recheck_b_v3.py)
- [`recheck_b_v4.py`](recheck_b_v4_recordB/recheck_b_v4.py) (engine in [2026-10-16_cleanroom_pooled_repro](../2026-10-16_cleanroom_pooled_repro/))

Result files the index names:

- `recheck_b_v4_out_B_2026-10-21_175838.json` (not copied: JSON outside results/ and config/)

## Files here

6 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-21_opt3_results/` at commit `85443b4d381767990125e7eed5e75a9053126606` of our internal repository. 23 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
