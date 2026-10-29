# 2026-10-12_plantslip_heldout

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does planting the simulator's text-editor formatting slips into real runs predict the real-minus-simulated catch gap of tool-output attack detectors that no earlier result had reported (row 2 of the approved experiment list)?

## What we did

Committed predictions and three baselines, then scored 15 previously unreported fixed-layer versions of the first-phase attack detector recipe, read at the end of each tool output, on three reader models (Qwen3-32B, Olmo-3.1-32B, Llama-3.3-70B) over 939 cells: x = catch lost when the slips are planted into second real runs, y = real minus simulated catch gap.

## What we found

Met the pre-set criterion: planting's forecast was off by 1.38 points on average vs 5.77 for the historical forecast (difference +4.40 [+2.63, +4.98]); rank correlation 0.93; the 15 settings come from 3 readers, so they are not independent. Realism-probe baseline rank correlation 0.67.

## In the post?

Not in the post.

## Status

Exploratory (descriptive)

## Builds on

[2026-10-12_plantslip_transfer](../2026-10-12_plantslip_transfer/); [2026-10-12_q2_olmo_reader](../2026-10-12_q2_olmo_reader/); [2026-10-12_q2_llama_reader](../2026-10-12_q2_llama_reader/); [2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/)

## Main scripts

- [`make_predictions.py`](make_predictions.py)
- [`make_manifest_part1.py`](make_manifest_part1.py)
- [`q2_x.py`](q2_x.py)
- [`q2_y.py`](q2_y.py)
- [`q2_b2.py`](q2_b2.py)
- [`q2_score_frozen_qwen3.py`](q2_score_frozen_qwen3.py)
- [`q2_score_frozen_olmo.py`](q2_score_frozen_olmo.py)
- [`q2_score_frozen_llama.py`](q2_score_frozen_llama.py)
- [`check_heldout_inputs.py`](check_heldout_inputs.py)

Result files the index names:

- `b2_2026-10-12_214616.json` (not copied: result left out to keep the release under 120 MB)
- [`heldout_inputs_check_2026-10-13_012403.txt`](results/heldout_inputs_check_2026-10-13_012403.txt)
- `q2_y_2026-10-12_220622.json` (not in this folder at the source commit)

## Files here

22 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-12_plantslip_heldout/` at commit `f4bb55be361643cd42a1579034ae52464ee31a3a` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
