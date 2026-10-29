# 2026-10-14_q2_item3_sidetask_check

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Where does the generic phrase "side task" appear in the system prompts of the prepared probe inputs, and could probes see it?

## What we did

A Modal CPU scan reporting only counts and hashes of the inputs prepared for the weak-simulator positive control (GPT-6-Astra simulating with no reference run and no codebase access), where an earlier scan had found the phrase, and of every other prepared folder not sealed for blind use, comparing system-prompt hashes across groups of runs.

## What we found

The phrase sits in 2 of 198 rows in each attack arm of the separate 18-environment LinuxArena cohort (1 of 100 in the planted arm), from the main task's honest system prompt, and in no row elsewhere (sidetask_check_POSTHOC_2026-10-14_102040.json).

## In the post?

Not in the post.

## Status

Current

## Builds on

The scan of what the probe sees (on a side branch)

## Main scripts

- [`run_sidetask_check.py`](run_sidetask_check.py)

Result files the index names:

- `sidetask_check_POSTHOC_2026-10-14_102040.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_q2_item3_sidetask_check/` at commit `a7f8ce90ded88d06cfa661936dc834d326d88ce8` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
