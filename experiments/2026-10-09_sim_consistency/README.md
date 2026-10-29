# 2026-10-09_sim_consistency

2026-10-09 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Does the simulated machine stay consistent with itself over a run (a file written then read back, created then listed)?

## What we did

Script count, no judge, of checkable write-then-read, create/delete-then-list and same-command-twice pairs in GPT-5.6-Sol Phase 1 attack runs (real run 1 vs simulated) and Qwen3.5-27B strict-rerun runs (real vs both simulators), with real runs as the negative control and planted inconsistencies as the positive control.

## What we found

No self-contradiction found in any simulated run and 5 of 5 planted inconsistencies caught, but 0 of 78 write-then-read pairs still allows a true rate up to about 4%.

## In the post?

Not in the post.

## Status

Current

## Builds on

Phase 1 GPT-5.6-Sol cohorts; [2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (strict rerun)

## Main scripts

- [`check_consistency.py`](check_consistency.py)

Result files the index names:

- [`consistency_2026-10-09_183356.json`](results/consistency_2026-10-09_183356.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-09_sim_consistency/` at commit `9ee28a78b8a7f56d883067962e55a53e45d9afed` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
