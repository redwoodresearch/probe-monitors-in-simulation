# 2026-10-15_il_analysis_09

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

When a third real attack run and a second simulated attack run are launched together in one interleaved launch, is their gap like the main estimate (removing effects of separate launch batches)?

## What we did

Ran the pre-registered analysis of the interleaved launch on top of the scale-up "error to add" job, plus post hoc add-ons (per-action results and a pooled root-mean-square gap) and checks of one pilot task (task 83) whose second-batch real honest run was missing from some score files; Modal CPU.

## What we found

Interleaved-launch analysis and extended records written; values not read for this index.

## In the post?

Not in the post.

## Status

Exploratory (interleaved-launch design not in the post)

## Builds on

[2026-10-15_q3_interleaved_run3_sim2](../2026-10-15_q3_interleaved_run3_sim2/)

## Main scripts

- [`run_il_analysis.py`](run_il_analysis.py)
- [`run_il_extended.py`](run_il_extended.py)
- [`run_il_posthoc.py`](run_il_posthoc.py)

Result files the index names:

- `il_analysis_real_final_2026-10-15_130350.json` (not copied: result file 1 MB or larger)
- `il_ext_analysis_real_final_2026-10-15_134705.json` (not copied: result file 1 MB or larger)
- `pooled_rms_gate_summary_2026-10-15_190417.json` (not copied: strings over 300 characters)

## Files here

13 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_il_analysis_09/` at commit `18eea9cf3d3bad98a2ad5a5c9e4e5a1f70424ff1` of our internal repository. 99 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
