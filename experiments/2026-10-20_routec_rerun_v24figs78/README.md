# 2026-10-20_routec_rerun_v24figs78

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does every drawn element of the per-task figures (Figures 6 and 7) map to a clean-room value?

## What we did

Mapped each plotted element of the per-task scatter and R-squared figures to the clean-room anchor rows, with jq gates, toy fixtures and Codex rounds; one Modal launch.

## What we found

Launch exited 0 with one map record and no STOP lines (launch summary); figure eligibility later ruled by the analysis owner.

## In the post?

Not in the post (verifies Appendix H Figures 6-7).

## Status

Current

## Builds on

[2026-10-16_cleanroom_pooled_repro](../2026-10-16_cleanroom_pooled_repro/) (anchor values); [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/)

## Main scripts

- [`map_figs78_a1.py`](map_figs78_a1.py)
- [`gate_figs78_prelim.jq`](gate_figs78_prelim.jq)
- [`gate_figs78_out.jq`](gate_figs78_out.jq)

Result files the index names:

- `figs78_map_2026-10-21_033233.json` (not copied: result file 1 MB or larger)
- [`figs78_gates_2026-10-21_033233.txt`](results/figs78_gates_2026-10-21_033233.txt)

## Files here

9 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-20_routec_rerun_v24figs78/` at commit `854f901b606cd8b44f30f87c263b441e7101a11f` of our internal repository. 17 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
