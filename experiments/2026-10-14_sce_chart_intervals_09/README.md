# 2026-10-14_sce_chart_intervals_09

2026-10-14 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

For the project lead's three-row chart of the earlier phase's mixed probe (trained on simulated attacks against real honest runs; Qwen3-32B reading GPT-5.6-Sol's runs), how large is the simulated-minus-real gap with paired intervals, before and after repairing the simulator's outputs?

## What we did

An independent recheck recomputed, from stored cross-validated scores with its own code and a check against the published catch rates, the gap for (1) the original probe on unrepaired simulated attacks, (2) the same probe on the 939 simulated attacks with two repairs applied and (3) a probe retrained on repaired runs, with 10,000 main-task resamples.

## What we found

Gap at 5%: as-is +57.6 points (95% +52.9 to +61.6); frozen probe on repaired runs +18.0 (95% +14.3 to +20.8); retrained on repaired runs +32.6 (95% +28.2 to +35.5) (sce_chart_2026-10-14_110423.printout.txt).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

2026-10-13 sce reply-probe scores

## Main scripts

- [`run_sce_chart_modal.py`](run_sce_chart_modal.py)
- [`sce_chart_boot.py`](sce_chart_boot.py)

Result files the index names:

- [`sce_chart_2026-10-14_110423.json`](results/sce_chart_2026-10-14_110423.json)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_sce_chart_intervals_09/` at commit `baa85a22e1faae44f89a86c770b849e0aadf6888` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
