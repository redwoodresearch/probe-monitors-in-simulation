# 2026-10-08_q3_swap_cue

2026-10-08 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Is the Phase 1 tool-output attack detector's real-versus-simulated shortfall caused by how the simulator renders text-editor outputs (insert template, line-number gutter, trailing newline)?

## What we did

Scored re-rendered simulated GPT-5.6-Sol attack runs (100 runs, then a 206-task extension) with the frozen Qwen3-32B tool-output detector, plus controls: layer 48 and the 32k cut, mechanism checks (what moved, dose response, locality), a TF-IDF text detector, recalibration without the batch of the original 200 real-honest reference runs (which overlap the compared cohort), scoring of the surface-edit, placebo and year-swap stagings, and 2026-10-11 post hoc Phase 1 checks.

## What we found

Simulated minus real catch at 5% was -20.5 [-26.1, -14.5]; both fixes raised it +20.4 [+16.4, +24.4], leaving -0.1 versus real; at the runner-up layer the direction flips (+9.9); an entry in our experiment log keeps -20.5 as an exploratory lead. README 2026-10-11: reading replies the detector catches 10-14 points fewer simulated attacks, no difference on commands.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/); [2026-10-07_q2_rendering_swap](../2026-10-07_q2_rendering_swap/); [2026-10-08_q2_surface_edit_sweep](../2026-10-08_q2_surface_edit_sweep/)

## Main scripts

- [`run_score_swap.py`](run_score_swap.py)
- [`run_score_swap_arms.py`](run_score_swap_arms.py)
- [`run_score_swap_frozen.py`](run_score_swap_frozen.py)
- [`analyze_swap_cue.py`](analyze_swap_cue.py)
- [`analyze_swap_paired.py`](analyze_swap_paired.py)
- [`analyze_dose_response.py`](analyze_dose_response.py)
- [`analyze_locality.py`](analyze_locality.py)
- [`run_tfidf_outputs.py`](run_tfidf_outputs.py)
- [`run_refit_no_base200.py`](run_refit_no_base200.py)
- [`run_fix939_stage.py`](run_fix939_stage.py)

Result files the index names:

- [`swap_cue_2026-10-08_014903.json`](results/swap_cue_2026-10-08_014903.json)
- `swap_frozen_2026-10-08_044006.json` (not copied: result left out to keep the release under 120 MB)
- `swap_paired_summary_2026-10-08_113500.jsonl` (not copied: result file type not copied (.jsonl))
- `swap_ext_summary_2026-10-08_130752.jsonl` (not copied: result file type not copied (.jsonl))
- [`dose_response_2026-10-08_145635.json`](results/dose_response_2026-10-08_145635.json)
- [`locality_2026-10-08_144403.json`](results/locality_2026-10-08_144403.json)
- `surface_edits_2026-10-08_175911.json` (not in this folder at the source commit)
- `fix939_POSTHOC_2026-10-11_020519.json` (not copied: result left out to keep the release under 120 MB)

## Files here

28 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-08_q3_swap_cue/` at commit `9db17c3db3775bc3c366cba53d9e06a7199209f1` of our internal repository. 81 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
