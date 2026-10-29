# 2026-10-12_q3_same_layer_table

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Is the contrast "readouts of the agent's commands agree between real and simulated runs, readouts of tool outputs do not" about the readout or about the layer?

## What we did

A \$0 reanalysis of the in-domain attack detector's stored scores: the command readout (end of each tool call) and the tool-output readout (end of each output) at layers 48 and 63 on a matched 681-cell cohort, with thresholds redrawn in every bootstrap draw and planted shifts of plus or minus 10 points.

## What we found

README at 5%: reply layer 63 -17.8 points [-21.7, -13.7], while command layer 48 +0.7, command layer 63 +0.4 and reply layer 48 +5.9 [+2.1, +9.4]; interaction -23.5 [-29.4, -17.5]: the break is specific to the reply readout at layer 63.

## In the post?

Not in the post.

## Status

Exploratory (post hoc)

## Builds on

[2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`analyze_same_layer.py`](analyze_same_layer.py)
- [`analyze_fix939_l48.py`](analyze_fix939_l48.py)
- [`null_swap_POSTHOC.py`](null_swap_POSTHOC.py)

Result files the index names:

- `same_layer_POSTHOC_2026-10-12_010722.json` (not in this folder at the source commit)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_same_layer_table/` at commit `208766897a4a3bd8f658e0e6ab3d1b1c50166490` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
