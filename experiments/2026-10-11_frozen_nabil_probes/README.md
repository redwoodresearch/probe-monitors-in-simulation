# 2026-10-11_frozen_nabil_probes

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Do frozen external probes (the AI Security Institute's public Apollo-style deception probes for Qwen3.5-27B, and an Apollo deception probe retrained earlier in this project) detect the project lead's hand-labelled sabotage commands, as an alternative to training a sabotage probe?

## What we did

Searched for a collaborator's TriviaQA-lie and APPS-backdoor probes (not found), froze a family of three deception probes and layer placements, scored stored Qwen3.5-27B readings at the end of each tool call on 40 first-stage labelled tasks (72 sabotage vs 889 other impactful commands) with planted checks and Holm correction for multiple tests, compared them on 182 tasks with the trained attack detector (a probe separating attack-instructed from honest Qwen3.5-27B runs), and ran a post hoc re-score averaging over command tokens.

## What we found

PLAN.md outcome: no family member passes; AISI layer 32 AUROC 0.639, catch 2.4% (fail); the other two failed pipeline checks; none Holm-significant; on the runs of the Qwen3.5-27B attack-detector comparison none matches the detector (AUROC 0.72-0.78 vs 0.998). COMMAND_MEAN.md: post hoc re-test on the same tasks.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_sabotage_stage1_labels](../2026-10-11_sabotage_stage1_labels/); [2026-10-11_generic_contrast_probe](../2026-10-11_generic_contrast_probe/); [2026-10-11_plant_design_audit](../2026-10-11_plant_design_audit/)

## Main scripts

- [`build_rows.py`](build_rows.py)
- [`score_frozen.py`](score_frozen.py)
- [`analyze_frozen.py`](analyze_frozen.py)
- [`score_part2_generic.py`](score_part2_generic.py)
- [`simulate_power_holm.py`](simulate_power_holm.py)
- [`yardstick_frozen.py`](yardstick_frozen.py)
- [`run_command_mean.py`](run_command_mean.py)
- [`analyze_command_mean.py`](analyze_command_mean.py)
- [`run_command_endpoint.py`](run_command_endpoint.py)

Result files the index names:

- `part1_result_2026-10-11_212326.json` (not copied: result left out to keep the release under 120 MB)
- `part2_compare_2026-10-11_211839.json` (not copied: result left out to keep the release under 120 MB)
- `power_holm_2026-10-11_210531.json` (not copied: strings over 300 characters)
- `yardstick_2026-10-12_000857.json` (not copied: result left out to keep the release under 120 MB)
- `command_mean_2026-10-12_113039.json` (not copied: result left out to keep the release under 120 MB)
- `command_endpoint_2026-10-12_131107.json` (not copied: result left out to keep the release under 120 MB)

## Files here

12 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_frozen_nabil_probes/` at commit `072f9fa7ca93ba108bf8d532c89e43660071891e` of our internal repository. 18 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
