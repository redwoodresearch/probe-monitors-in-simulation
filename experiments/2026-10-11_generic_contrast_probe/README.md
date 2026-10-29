# 2026-10-11_generic_contrast_probe

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does a sabotage probe built only from generic contrast statement pairs (as in "Simple probes can catch sleeper agents"), using no attack text, detect the project lead's hand-labelled sabotage commands, and does it match the trained attack detector on that detector's runs?

## What we did

Computed contrast directions on Qwen3.5-27B from 64 freshly written statement pairs (one H200 GPU), then scored stored readings at the end of each tool call: part 1 on 40 first-stage tasks with judge-panel labels against permutation, random-direction and planted checks; part 2 against the attack detector (a probe separating attack-instructed from honest Qwen3.5-27B runs); about \$0.31.

## What we found

Commit c6ff901f4: pre-registered direction A (layer 48) fails, weighted AUROC 0.604 [0.494, 0.709], catch at 1% 0.000; exploratory direction B (layer 46) failed its pipeline checks; part 2 comparison in part2_compare_2026-10-11_200530.json.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_sabotage_stage1_labels](../2026-10-11_sabotage_stage1_labels/); the clean attack-detector test's runs ([2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/))

## Main scripts

- [`run_contrast.py`](run_contrast.py)
- [`run_contrast_b.py`](run_contrast_b.py)
- [`score_part1.py`](score_part1.py)
- [`score_part2.py`](score_part2.py)
- [`analyze_part1.py`](analyze_part1.py)
- [`analyze_part2.py`](analyze_part2.py)
- [`analyze_direction_b.py`](analyze_direction_b.py)
- [`simulate_power.py`](simulate_power.py)
- [`verify_part2_frozen.py`](verify_part2_frozen.py)
- [`check_frozen_direction_hash.py`](check_frozen_direction_hash.py)

Result files the index names:

- [`part2_compare_2026-10-11_200530.json`](results/part2_compare_2026-10-11_200530.json)
- [`layer_2026-10-11_195822.json`](results/layer_2026-10-11_195822.json)
- [`direction_b_layer_2026-10-11_205421.json`](results/direction_b_layer_2026-10-11_205421.json)
- `manifest_part1_result_v1.json` (not copied: strings over 300 characters)
- `part2_frozen_direction_verify_2026-10-12_025457.json` (not copied: result left out to keep the release under 120 MB)

## Files here

10 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_generic_contrast_probe/` at commit `5bac7a5f8f58144908459ae156c4383ecb310dab` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
