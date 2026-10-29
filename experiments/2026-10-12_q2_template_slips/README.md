# 2026-10-12_q2_template_slips

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Which editor-format slips survive each reader's chat template, and does Qwen3.5-27B's frozen reply detector react when a surviving slip is planted into real runs?

## What we did

Step 1 (\$0.20): built paired versions differing in one component and checked whether rendered prompts and readout tokens differ under Qwen3-32B, Qwen3.5-27B and Olmo. Step 2 (\$13.16): planted the gutter strip or the note drop into all 939 second real runs, re-extracted Qwen3.5 activations and scored with the frozen detector.

## What we found

README: the trailing newline never reaches Qwen3.5 (0 pairs survive its template); every other component survives both templates; neither planted slip moves Qwen3.5's detector at 5% ("does not respond", interval within plus or minus 2 points).

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-12_q2_qwen35_reply_readout](../2026-10-12_q2_qwen35_reply_readout/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`run_template_slips.py`](run_template_slips.py)
- [`run_slip_readouts.py`](run_slip_readouts.py)
- [`run_plant_stage.py`](run_plant_stage.py)
- [`run_score_plants.py`](run_score_plants.py)
- [`analyze_plants.py`](analyze_plants.py)
- [`check_scoring_gates.py`](check_scoring_gates.py)

Result files the index names:

- `template_slips_2026-10-12_122633.json` (not in this folder at the source commit)
- `slip_readouts_2026-10-12_122620.json` (not in this folder at the source commit)
- `plant_reading_2026-10-12_134340.json` (not in this folder at the source commit)
- [`plant_score_full_2026-10-12_134312.json`](results/plant_score_full_2026-10-12_134312.json)

## Files here

11 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-12_q2_template_slips/` at commit `f339aa20baf57f6baaed36a39fc1284552f3630c` of our internal repository. 17 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
