# 2026-10-06_q3_attack_detector

2026-10-06 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does an in-domain attack detector (a probe trained on real attack runs against real honest runs) catch simulated attacks as often as real ones?

## What we did

Phase 1: Qwen3-32B probes at call ends and output ends, trained on GPT-5.6-Sol's real attack vs honest runs, scored on real and simulated attack runs at thresholds from real honest runs; then a fully simulated evaluation with thresholds from 437 simulated honest runs, and a retrain on cleaned text (all seven cumulative cleanups: times and dates, numbers and IDs, whitespace, harness format, diff lines, empty diff lines, editor output format).

## What we found

README status: at 5% false alarms on tool calls it catches 61% of real and 64% of simulated attacks; on tool outputs 15 points fewer simulated; fully simulated tool-output gap -14.2 [-22.5, -4.7], shrinking to -2.6 [-13.5, +7.2] on cleaned text.

## In the post?

Not in the post.

## Status

Current (Phase 1 record); the Phase 1 forerunner of the post's off-policy probe idea, later rebuilt with Qwen3.5-27B as the reader

## Builds on

[2026-10-06_all_layers_extraction](../2026-10-06_all_layers_extraction/); [2026-10-06_astra_sim_honest](../2026-10-06_astra_sim_honest/)

## Main scripts

- [`run_q3_attack.py`](run_q3_attack.py)
- [`run_score_sim_honest.py`](run_score_sim_honest.py)
- [`run_null_selection.py`](run_null_selection.py)
- [`analyze_q3_attack.py`](analyze_q3_attack.py)
- [`analyze_sim_honest.py`](analyze_sim_honest.py)
- [`plot_q3_attack.py`](plot_q3_attack.py)

Result files the index names:

- `metrics_2026-10-06_181548__2026-10-06_182739.json` (not copied: result left out to keep the release under 120 MB)
- [`null_selection_2026-10-07_102220.json`](results/null_selection_2026-10-07_102220.json)
- [`sim_honest_2026-10-07_134849__2026-10-07_135201.json`](results/sim_honest_2026-10-07_134849__2026-10-07_135201.json)

## Files here

13 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-06_q3_attack_detector/` at commit `f5e3a446a986f570768c8637d86fc1b4c5e9bf92` of our internal repository. 59 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
