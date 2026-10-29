# 2026-10-12_q2_llama_plant

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Gate row 3: if the simulator's two editor-format slips are planted into Llama-3.3-70B's second real runs, does its frozen reply detector lose catch by the predicted amount?

## What we did

Extracted Llama-3.3-70B activations for 845 planted second real runs and 20 controls, scored them on one Modal CPU container with the frozen part-1 detector at all 7 layers (gate: controls reproduce stored scores), and compared catch before and after planting.

## What we found

Catch falls from 80.3% to 75.7% at 5%, -4.6 points [-6.2, -3.2] (-5.5 [-6.8, -4.4] at 1%), against a committed prediction of -6.5 plus or minus 3.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_q2_llama_reader](../2026-10-12_q2_llama_reader/); [2026-10-12_plantslip_transfer](../2026-10-12_plantslip_transfer/)

## Main scripts

- [`run_score_plant_llama.py`](run_score_plant_llama.py)
- [`analyze_plant_llama.py`](analyze_plant_llama.py)

Result files the index names:

- [`llama_plant_delta_2026-10-12_212724.json`](results/llama_plant_delta_2026-10-12_212724.json)
- [`llama_plant_scores_planted-e2_2026-10-12_212056.json`](results/llama_plant_scores_planted-e2_2026-10-12_212056.json)
- [`llama_plant_scores_controls-e2_2026-10-12_203523.json`](results/llama_plant_scores_controls-e2_2026-10-12_203523.json)

## Files here

2 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-12_q2_llama_plant/` at commit `acd7a48b0d1d6df4a420dbc09ed33a700c86e1b3` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
