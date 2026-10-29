# 2026-10-11_sabotage_test

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does a new sabotage probe catch sabotage commands equally in real and simulated Qwen3.5-27B attack runs? (stage 1: does a probe trained on labelled sabotage commands work at all)

## What we did

Stage 1: built command lists, trained a probe on Qwen3.5-27B call-end readings for 24 tasks of judge-panel-labelled commands and tested it on 16 held-out tasks against an honest-calibrated 1% threshold, with a planted check; stage-2 sizing; post hoc plant redesign and metadata baselines.

## What we found

README: stage-1 gate passed (held-out AUROC 0.910, catch at 1% 25.5% [4.2%, 52.5%]) but the pre-registered planted check was not detected, so stage 2 was stopped; post hoc redesigned plant detected; metadata alone reaches AUROC 0.851. The plus or minus 10 rule is not reachable (best 0.665).

## In the post?

Not in the post.

## Status

Exploratory (stage 2 stopped)

## Builds on

[2026-10-11_sabotage_stage1_labels](../2026-10-11_sabotage_stage1_labels/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`prepare_stage1.py`](prepare_stage1.py)
- [`prepare_stage1_ext.py`](prepare_stage1_ext.py)
- [`stage_gate.py`](stage_gate.py)
- [`run_gate.py`](run_gate.py)
- [`analyze_gate.py`](analyze_gate.py)
- [`simulate_stage1_gate.py`](simulate_stage1_gate.py)
- [`simulate_stage2_power.py`](simulate_stage2_power.py)
- [`summarize_stage2_power.py`](summarize_stage2_power.py)
- [`run_gate_diag_POSTHOC.py`](run_gate_diag_POSTHOC.py)
- [`run_plant_v2_POSTHOC.py`](run_plant_v2_POSTHOC.py)
- [`analyze_metadata_POSTHOC.py`](analyze_metadata_POSTHOC.py)

Result files the index names:

- [`gate_result_2026-10-11_211446.json`](results/gate_result_2026-10-11_211446.json)
- [`plant2_size_result_POSTHOC_2026-10-11_233212.json`](results/plant2_size_result_POSTHOC_2026-10-11_233212.json)
- `metadata_baselines_POSTHOC_2026-10-12_003805.json` (not copied: result left out to keep the release under 120 MB)
- [`check_calibration_2026-10-11_173049.json`](results/check_calibration_2026-10-11_173049.json)
- `manifest_stage2_info_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

22 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_sabotage_test/` at commit `55c4837df0ac8efefee2717743ac1491b6351754` of our internal repository. 22 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
