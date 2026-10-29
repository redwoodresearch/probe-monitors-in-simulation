# 2026-10-15_loro_sep_09

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Under separate thresholds, is the tool-output offset of the +311 added tasks caused by simulation or by which launch the real runs came from?

## What we did

Separate-threshold leave-one-recipe-out, a launch check (sim vs run 1, sim vs run 2, run 2 vs run 1), a same-attack pool-swap check and run-length caps, on pooled 455-task per-draw arrays; v2/v3 reruns after input corrections.

## What we found

Records written for command and reply sides; values not read for this index.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-15_loro_09](../2026-10-15_loro_09/); [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/)

## Main scripts

- [`run_loro_sep.py`](run_loro_sep.py)
- [`run_launch_check_v3.py`](run_launch_check_v3.py)
- [`run_sameattack_check_v2.py`](run_sameattack_check_v2.py)
- [`run_loro_sep_runlen.py`](run_loro_sep_runlen.py)

Result files the index names:

- `launch_check311v3_CR_2026-10-15_162311.json` (not copied: strings over 300 characters)
- `loro_runlen_cap30_CR_2026-10-15_165508.json` (not copied: strings over 300 characters)

## Files here

15 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_loro_sep_09/` at commit `fafe86930332cd9c003cf4f68357bd5bc9dd7343` of our internal repository. 51 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
