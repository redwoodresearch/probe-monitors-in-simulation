# 2026-10-14_score_shift_route_09

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Would modelling simulated attack scores as a shifted (and rescaled) copy of real ones give tighter, still valid simulated-minus-real catch bands than estimating the difference directly?

## What we did

For the Sol-trained probe on 177 pilot tasks, compared the direct catch difference with location and location-plus-scale shift models (task bootstrap, thresholds on the second real honest arm), checking the shift assumption with quantile plots.

## What we found

At 1% (honest-B threshold) the direct estimate is −0.85 points (95% −7.46 to +5.71, width 13.2) and the location model 0.00 (95% −3.93 to +5.28, width 9.2), with the shift assumption holding (score_shift_2026-10-14_121911.printout.txt). The post reports direct differences.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_priority1_firstcut](../2026-10-14_priority1_firstcut/)

## Main scripts

- [`run_score_shift_modal.py`](run_score_shift_modal.py)
- [`score_shift.py`](score_shift.py)

Result files the index names:

- [`score_shift_2026-10-14_121911.json`](results/score_shift_2026-10-14_121911.json)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_score_shift_route_09/` at commit `a0efcf6f69e9cee36c6e91f0ce4d12a2afd89cc3` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
