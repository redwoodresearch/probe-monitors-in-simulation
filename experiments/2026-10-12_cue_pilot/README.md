# 2026-10-12_cue_pilot

2026-10-12 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Does the frozen evaluation-awareness probe (Qwen3.5-27B, layer 32) notice an explicit "This is an evaluation." sentence planted into real honest LinuxArena runs, compared with probes aimed at the cue's tokens at layers 2 and 4?

## What we did

On one H200, planted the cue or a neutral sentence near or far in 120 real honest runs (20 test, 60 calibration, 40 training), scored them with the frozen layer-32 probe and newly fit token-level aimed probes at layers 2 and 4, each with 20 label-swap null refits. Billed \$1.87.

## What we found

Screen result: the frozen layer-32 probe caught 0 of 20 cue runs (far and near), reading "L32's readout or training doesn't carry this cue"; the aimed probes passed family-wise.

## In the post?

Not in the post (the post's evaluation-awareness footnote uses the later [2026-10-14_p2_evalaware_nguyen](../2026-10-14_p2_evalaware_nguyen/) probe).

## Status

Exploratory (followed by [2026-10-12_matched_training_d](../2026-10-12_matched_training_d/))

## Builds on

[2026-10-10_eval_awareness_probe](../2026-10-10_eval_awareness_probe/); [2026-10-11_is_this_a_test](../2026-10-11_is_this_a_test/)

## Main scripts

- [`run_cue_pilot.py`](run_cue_pilot.py)
- [`analyze_cue_pilot.py`](analyze_cue_pilot.py)
- [`analyze_POSTHOC_l32_shift.py`](analyze_POSTHOC_l32_shift.py)

Result files the index names:

- `screen_2026-10-12_153045.json` (not copied: result left out to keep the release under 120 MB)
- [`full_2026-10-12_151458.json`](results/full_2026-10-12_151458.json)
- [`l32_shift_POSTHOC_2026-10-12_153138.json`](results/l32_shift_POSTHOC_2026-10-12_153138.json)

## Files here

3 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-12_cue_pilot/` at commit `cfb6b065caab2cc9aa3b8b7861cc21ca4ab605b1` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
