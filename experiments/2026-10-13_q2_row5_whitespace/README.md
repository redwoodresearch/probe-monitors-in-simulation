# 2026-10-13_q2_row5_whitespace

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does planting the simulated runs' whitespace habits into real runs move the last-token tool-output attack detector at layer 48 toward the simulated runs' catch rate?

## What we did

Planted simulated-style whitespace into GPT-5.6-Sol's second real attack runs (936 runs changed), re-extracted Qwen3-32B and scored with the frozen in-domain last-token detector at layer 48 on the 939 paired task cells. Modal GPU.

## What we found

README (delta_ws_2026-10-13_011017.json): catch falls 63.05% to 60.41%, -2.64 [-3.69, -1.56] at 5% and -2.34 at 1%, the opposite direction from the simulated-minus-real gap (+4.15 [+0.90, +7.31]); prediction +0.5 +/- 2 missed.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_q2_pinned_layers](../2026-10-12_q2_pinned_layers/) (row 1); the earlier attack detector

## Main scripts

- [`stage_whitespace.py`](stage_whitespace.py)
- [`score_whitespace.py`](score_whitespace.py)
- [`delta_whitespace.py`](delta_whitespace.py)

Result files the index names:

- [`delta_ws_2026-10-13_011017.json`](results/delta_ws_2026-10-13_011017.json)
- [`scores_ws_planted_2026-10-13_010958.json`](results/scores_ws_planted_2026-10-13_010958.json)
- [`scores_ws_controls_2026-10-13_004442.json`](results/scores_ws_controls_2026-10-13_004442.json)

## Files here

3 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-13_q2_row5_whitespace/` at commit `6c42a9526e68fc6e0686c5e8a20be4cfd6a2cdb8` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
