# 2026-10-12_q3_simtrained_range

2026-10-12 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

How sure are we that a command monitor trained only on simulated attack runs works the same on real runs?

## What we did

\$0, stored refit draws: 95% range of real minus simulated catch at 1% for the simulation-trained monitor in the acting set-up (196 tasks, 500 refits) and the reading set-up (530 tasks), threshold carried over or re-set; label-shuffle null added after review.

## What we found

README: acting -0.23 points [-2.68, +2.51], reading -0.53 [-1.27, +0.84] with the threshold carried over; re-set thresholds -1.97 [-10.87, +8.15] and -0.19 [-5.08, +5.17].

## In the post?

Not in the post.

## Status

Exploratory (post hoc); its yardstick row filled by [2026-10-12_q3_simtrained_yardstick](../2026-10-12_q3_simtrained_yardstick/)

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/); [2026-10-08_impactful_refit_bootstrap](../2026-10-08_impactful_refit_bootstrap/)

## Main scripts

- [`analyze_simtrained_range.py`](analyze_simtrained_range.py)
- [`run_plants_simtrained_POSTHOC.py`](run_plants_simtrained_POSTHOC.py)

Result files the index names:

- `simtrained_range_2026-10-12_001352.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_simtrained_range/` at commit `b973ff3589394d74f29cffcdf6bca74faba917cb` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
