# 2026-10-09_q2_deception_commands

2026-10-09 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does the frozen deception probe (Apollo's recipe retrained on Qwen3.5-27B) score the project lead's hand-labelled sabotage commands above benign commands from the same real attack runs?

## What we did

Under a plan frozen before scoring, scored 60 keyed commands (30 sabotage, 30 benign, labelled blind) at Qwen3.5-27B command tokens with the step-2 probe (layer 31), from stored activations only.

## What we found

Separation not established; sabotage minus benign +0.06 honest standard deviations [-0.49, +0.53], ranking score 0.52, 3 of 30 sabotage vs 0 of 30 benign crossed the alarm line (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-09_q2_deception_apollo](../2026-10-09_q2_deception_apollo/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (the project lead's 60 keyed commands)

## Main scripts

- [`run_check_and_score.py`](run_check_and_score.py)
- [`analyze_sabotage_scores.py`](analyze_sabotage_scores.py)
- [`check_honest_readouts.py`](check_honest_readouts.py)

Result files the index names:

- [`sabotage_scores_analysis_2026-10-10_210808.json`](results/sabotage_scores_analysis_2026-10-10_210808.json)
- [`compatibility_check_2026-10-10_210326.json`](results/compatibility_check_2026-10-10_210326.json)
- [`honest_readouts_check_2026-10-10.json`](results/honest_readouts_check_2026-10-10.json)

## Files here

3 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-09_q2_deception_commands/` at commit `d866a9474c14d16a0b7499e5fec71e545a1e8e45` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
