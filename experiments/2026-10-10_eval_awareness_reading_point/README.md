# 2026-10-10_eval_awareness_reading_point

2026-10-10 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Does the evaluation-awareness probe's weak response to planted cues come from distance to the cue, and does a probe at a fixed reading point (assistant-start token, layer 4) respond specifically to the cue?

## What we did

A free cue-distance analysis on stored rows (1,862 commands in 99 run pairs) and a retrain at a fixed reading point with a frozen neutral-line gate on 20 fresh tasks (the project lead's own run).

## What we found

README: partial attenuation with distance; the confirmation gate fails because a neutral line also moved scores +1.604 [1.286, 1.944] against the checker cue's +8.164, so the response is not specific to the cue; the real-versus-simulated stage was not launched.

## In the post?

Not in the post.

## Status

Current (gate failed; direction dropped)

## Builds on

[2026-10-10_eval_awareness_probe](../2026-10-10_eval_awareness_probe/)

## Main scripts

- [`prepare_inputs.py`](prepare_inputs.py)
- [`stage_inputs.py`](stage_inputs.py)
- [`run_pilot.py`](run_pilot.py)
- [`run_guarded.py`](run_guarded.py)
- [`analyze_pilot.py`](analyze_pilot.py)
- [`plot_distance.py`](plot_distance.py)
- [`plot_confirmation.py`](plot_confirmation.py)

Result files the index names:

- `metrics.json` (not copied: result left out to keep the release under 120 MB)
- [`frozen_thresholds.json`](results/calibration_v1/frozen_thresholds.json)
- `early_probe_freeze.json` (not copied: result left out to keep the release under 120 MB)
- `final_cost_audit.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-10_eval_awareness_reading_point/` at commit `3b5a777fea1a06f652481dce546f4cce744443a4` of our internal repository. 19 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
