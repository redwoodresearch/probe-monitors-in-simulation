# 2026-10-13_like_for_like_D_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Re-expressed against a real-trained detector with the same number of training negatives, how far do the simulation-trained recipes (few real attacks added; world-direction removal) still trail real training?

## What we did

Recomputed D (second real run's catch under the cell minus under the count-matched real-trained detector, averaged over its 5 subsample draws) for the few-real-attacks and world-direction cells, Qwen3-32B tool-output probe reading GPT-5.6-Sol runs, layers 48 and 63, 1% and 5%, same 2,000 draws. Modal CPU, \$0.014.

## What we found

lfl_D_POSTHOC_2026-10-13_175001.printout.txt: at layer 48, 1%, the count-matched detector catches 28.6% vs 36.9% for the original; D for fully fixed simulated/simulated training goes from -16.0 [-22.5, -7.1] to -7.7 [-13.0, -0.6]; with 100 real attacks added -5.8 [-10.9, +1.5]; mixed-training world-direction variants reach about -0.2 to -4.6.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_few_real_attacks_POSTHOC](../2026-10-13_few_real_attacks_POSTHOC/); [2026-10-13_world_direction_POSTHOC](../2026-10-13_world_direction_POSTHOC/); [2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (count-matched cell a_m)

## Main scripts

- [`run_lfl_D.py`](run_lfl_D.py)

Result files the index names:

- [`lfl_D_POSTHOC_2026-10-13_175001.printout.txt`](results/lfl_D_POSTHOC_2026-10-13_175001.printout.txt)
- `manifest_lfl_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_like_for_like_D_POSTHOC/` at commit `856f83c8eaa49b38ca1405eb6ab9eb386226ab0d` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
