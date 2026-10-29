# 2026-10-11_start_condition_check

2026-10-11 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Do real and simulated runs start the same way (no simulated run forked from a prefix while real runs start fresh)?

## What we did

\$0 read-only check of every .eval in each cohort: fork metadata, seeded prefixes, and hashes of the system message, first user turn and whole first input at the acting model's first call.

## What we found

CHECK.md: in every cohort checked, real and simulated runs start the same way; all checked simulated runs are fresh starts and 5,652 runs reached the first call with byte-identical opening inputs between real and simulated runs of the same task.

## In the post?

Not in the post.

## Status

Current

## Builds on

Phase 1 and Phase 2 cohorts

## Main scripts

- [`check_start_conditions.py`](check_start_conditions.py)

Result files the index names:

- `manifest_start_conditions_v1.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_start_condition_check/` at commit `ec410b9956786162a9241ec7fa11122d816145cf` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
