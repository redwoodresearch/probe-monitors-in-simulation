# 2026-10-11_whitebox_dataset_probe

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does a sabotage probe trained on public white-box-control datasets (code sabotage, deception, rule violation) detect the project lead's labelled sabotage commands, and does it match the trained attack detector?

## What we did

Trained and froze a logistic probe on Qwen3.5-27B activations of 600 public dataset rows (layer 47), then scored stored readings at the end of each tool call: part 1 on 40 first-stage labelled tasks, part 2 on the 182-task set of the attack detector (a probe separating attack-instructed from honest Qwen3.5-27B runs); Modal, about \$0.93.

## What we found

README: validity gate passed (AUROC 1.0 in-distribution); part 1 fails (AUROC 0.396, below chance; catch at 1% 0.0); part 2 not consistent with the detector (AUROC 0.642, catch 0.5%; commit ede95db05).

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_sabotage_stage1_labels](../2026-10-11_sabotage_stage1_labels/); [2026-10-11_frozen_nabil_probes](../2026-10-11_frozen_nabil_probes/)

## Main scripts

- [`count_lengths.py`](count_lengths.py)
- [`run_whitebox.py`](run_whitebox.py)
- [`score_whitebox.py`](score_whitebox.py)
- [`analyze_whitebox.py`](analyze_whitebox.py)
- [`score_part2_generic.py`](score_part2_generic.py)

Result files the index names:

- `part1_2026-10-11_230956.json` (not copied: result left out to keep the release under 120 MB)
- `part2_2026-10-11_231018.json` (not copied: result left out to keep the release under 120 MB)
- [`dataset_lengths_2026-10-11.json`](results/dataset_lengths_2026-10-11.json)

## Files here

6 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_whitebox_dataset_probe/` at commit `629bd0dd050836679b72c3d1e0e4bc2425a74e2a` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
