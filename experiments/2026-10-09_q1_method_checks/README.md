# 2026-10-09_q1_method_checks

2026-10-09 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Method checks on the Phase 1 real-versus-simulated probe: how much do all seven cumulative cleanups (times and dates, numbers and IDs, whitespace, harness format, diff lines, empty diff lines, editor output format) change each world's text, does a wider regularization search with a selection-aware null change the result, and what do the cleanup ladder and planted checks look like in catch at 1% and 5% false alarms?

## What we did

Modal and local re-analyses of the Phase 1 Qwen3-32B real-versus-simulated probe: per-world counts of changes from all seven cleanups, a wide C grid with fitter convergence audit and single-column reproduction, and post hoc catch-at-false-alarm tables for the ladder and planted checks.

## What we found

All seven cleanups change 0.314 of real and 0.312 of simulated outputs. A wider C search raises the probe with all seven cleanups to 0.570 from 0.541. Catch at 1% drops from 0.62 with no text cleanup to 0.01 with all seven cleanups, the same as real versus real (post hoc).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-06_q1_segment_end](../2026-10-06_q1_segment_end/); [2026-10-10_q1_planted_checks](../2026-10-10_q1_planted_checks/)

## Main scripts

- [`run_level7_changes.py`](run_level7_changes.py)
- [`run_q1_cgrid.py`](run_q1_cgrid.py)
- [`run_repro_single_column.py`](run_repro_single_column.py)
- [`run_catch_ladder.py`](run_catch_ladder.py)
- [`analyze_catch_plants.py`](analyze_catch_plants.py)
- [`analyze_ladder_refresh.py`](analyze_ladder_refresh.py)
- [`analyze_levels_rule.py`](analyze_levels_rule.py)

Result files the index names:

- `level7_changes_2026-10-09_171437.json` (not in this folder at the source commit)
- `fitter_audit_q1_ladder_2026-10-10_002647.json` (not in this folder at the source commit)
- `ladder_refresh_2026-10-10_154139.json` (not in this folder at the source commit)
- `catch_ladder_POSTHOC_2026-10-10_224950.json` (not in this folder at the source commit)
- `catch_plants_POSTHOC_2026-10-10_224657.json` (not in this folder at the source commit)

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_q1_method_checks/` at commit `c96c61a695d30270037541b8453a5c89447b87df` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
