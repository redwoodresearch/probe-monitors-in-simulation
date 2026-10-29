# 2026-10-14_fpr_sweep_unit

2026-10-14 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

At fixed false-positive rates on real honest units (replies or commands), from 1 in 10,000 to 1 in 10, does each probe catch simulated attack runs as often as real ones, beside the gap between two real runs (the project manager's priority 2)?

## What we did

Unit-level and run-level threshold sweeps on stored per-row scores of Qwen3.5-27B's 183-task pilot cohort for the Sol-trained probe, FIBS deception, harmfulness, refusal and high-stakes probes plus a positive control, with environment-stratified task bootstraps; part (c) scored on-policy probes at pre-selected operating points; add-ons used the 1,344-run enlarged honest pool and a third panel.

## What we found

Of 90 readable points, simulated minus mean real was resolved upward at 13 and downward at 0, run 2 minus run 1 resolved downward at 6; at alarm levels flagging 5% or fewer of honest runs, no simulated-minus-real gap was resolved (readings_2026-10-15_015250.json).

## In the post?

Not in the post.

## Status

Superseded by [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) (the final pooled sweep on all tasks, Appendix C other alarm levels)

## Builds on

[2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/); [2026-10-14_p2_fibs_pair](../2026-10-14_p2_fibs_pair/); [2026-10-14_p2_refusal_harm](../2026-10-14_p2_refusal_harm/); [2026-10-14_p2_highstakes](../2026-10-14_p2_highstakes/); [2026-10-14_q3_q35_realhonest_1e4](../2026-10-14_q3_q35_realhonest_1e4/) (enlarged pool)

## Main scripts

- [`run_fprsweep_unit.py`](run_fprsweep_unit.py)
- [`run_fprsweep_run_a.py`](run_fprsweep_run_a.py)
- [`run_fprsweep_unit_addon.py`](run_fprsweep_unit_addon.py)
- [`run_panel3.py`](run_panel3.py)
- [`analyze_readings.py`](analyze_readings.py)
- [`analyze_sizing.py`](analyze_sizing.py)
- [`plot_fprsweep_unit.py`](plot_fprsweep_unit.py)

Result files the index names:

- `fprsweep_unit_2026-10-15_003643.json` (not copied: strings over 300 characters)
- `fprsweep_run_a_2026-10-15_011009.json` (not copied: strings over 300 characters)
- `fprsweep_unit_addon_2026-10-15_015319.json` (not copied: result left out to keep the release under 120 MB)
- `panel3_2026-10-15_021909.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `readings_2026-10-15_015250.json` (not copied: result left out to keep the release under 120 MB)
- `sizing_2026-10-15_015615.json` (not copied: strings over 300 characters)

## Files here

15 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_fpr_sweep_unit/` at commit `7cb75cb302a3f5a9d1d623f027526963890f022b` of our internal repository. 87 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
