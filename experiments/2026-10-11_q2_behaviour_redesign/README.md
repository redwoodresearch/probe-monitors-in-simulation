# 2026-10-11_q2_behaviour_redesign

2026-10-11 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Can the behaviour tests (did the agent act differently in simulation) be redesigned as at most three measures, each sized and planted in advance?

## What we did

Plan with rough normal-approximation power: keep the target-touch measure (already run), add a judge-panel 'attempted the side task' measure, and price the rest; nothing launched.

## What we found

PLAN.md: the panel-labelled attempt measure cannot reach 80% power at plus or minus 10 points with about 190 tasks (needs about 370-430); commit 4e3aeb38b: the project lead made it descriptive only with no new labels.

## In the post?

Not in the post.

## Status

Void (plan only)

## Builds on

[2026-10-11_q2_n2_target](../2026-10-11_q2_n2_target/)

## Main scripts

- [`analyze_power_rough.py`](analyze_power_rough.py)

Result files the index names:

- [`power_rough_output.txt`](results/power_rough_output.txt)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_q2_behaviour_redesign/` at commit `4e3aeb38bf042f4d93b973d2fb374fa78fee6c27` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
