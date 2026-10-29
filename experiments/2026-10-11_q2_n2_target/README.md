# 2026-10-11_q2_n2_target

2026-10-11 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Do Qwen3.5-27B's attack runs touch the side task's target as often in simulation as on real machines?

## What we did

\$0 on the version-3 target-touch labels with shared-only commands dropped: share of attack runs touching the target at least once, real run 1 vs simulated over 189 tasks, real run 2 as yardstick, Holm equivalence tests and a planted check run first.

## What we found

README: agree within plus or minus 10 points, real run 1 minus simulated +1.6 points [-3.2, +6.4] (15.9% vs 14.3%), real run 1 minus run 2 -2.6 [-6.7, +1.0]; per-run share agrees within 3 points; plants detected 200 of 200; preliminary until a fresh human check of the labels.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_target_touch_labels](../2026-10-11_target_touch_labels/); [2026-10-11_q2_zoo_plans](../2026-10-11_q2_zoo_plans/)

## Main scripts

- [`analyze_n2_target.py`](analyze_n2_target.py)

Result files the index names:

- [`n2_target_full_2026-10-11_140829.json`](results/n2_target_full_2026-10-11_140829.json)
- [`n2_target_full_2026-10-11_140041.json`](results/n2_target_full_2026-10-11_140041.json)
- [`n2_target_plant_2026-10-11_140012.json`](results/n2_target_plant_2026-10-11_140012.json)

## Files here

1 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_q2_n2_target/` at commit `2d1c377830887637dbc3485e5ca019df18ebf398` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
