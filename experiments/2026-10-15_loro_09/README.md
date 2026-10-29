# 2026-10-15_loro_09

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

For a new probe recipe, how much should we expect simulation to distort its measured catch rate?

## What we did

Leave-one-recipe-out prediction test over six probe families on the common-comparison per-draw table (183 then 455 tasks), estimating mean signed gap and between-recipe spread; Modal CPU.

## What we found

455 tasks, tool outputs, 1% run level: mean signed gap +0.69 points [-0.82, +2.20]; predicted range for a new recipe -5.4 to +6.8; 6 of 6 held-out recipes inside (loro455 JSON headline).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) perdraw455

## Main scripts

- [`run_loro.py`](run_loro.py)

Result files the index names:

- `loro_2026-10-15_105655.json` (not copied: strings over 300 characters)
- `loro455_2026-10-15_114303.json` (not copied: strings over 300 characters)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_loro_09/` at commit `96b6acb100ea27e4e215e1b830052389e215b716` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
