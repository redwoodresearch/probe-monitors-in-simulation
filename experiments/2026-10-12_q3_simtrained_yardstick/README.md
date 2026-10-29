# 2026-10-12_q3_simtrained_yardstick

2026-10-12 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

For the monitor trained on simulated runs, how much do two real runs differ (the real-vs-real yardstick)?

## What we did

Rescored the 196-task scale-up (191 tasks) with a simulation-trained to second-real-run cell over 500 refit draws on Modal, with a 10-point second-run plant.

## What we found

README: second minus first real catch -0.12 points [-3.28, +2.52] vs real minus simulated -0.23 [-2.68, +2.51]; plant detected; the false-alarm control fails its plus or minus 1-point criterion.

## In the post?

Not in the post.

## Status

Exploratory (post hoc)

## Builds on

[2026-10-12_q3_simtrained_range](../2026-10-12_q3_simtrained_range/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`run_plants_yardstick.py`](run_plants_yardstick.py)
- [`analyze_yardstick.py`](analyze_yardstick.py)
- [`spawn_score.py`](spawn_score.py)
- [`check_gate.py`](check_gate.py)

Result files the index names:

- `yardstick_2026-10-12_031021.json` (not copied: result left out to keep the release under 120 MB)
- [`gate_draw0_2026-10-12_013819.json`](results/gate_draw0_2026-10-12_013819.json)

## Files here

7 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-12_q3_simtrained_yardstick/` at commit `c09ea2f8baf98c9dec3000d92c6427502d245e9d` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
