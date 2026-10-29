# 2026-10-15_raw_eval_read_09

2026-10-15 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Is the empty first user turn in interleaved-launch donor runs real, with the task text in the system prompt?

## What we did

Read the structure (no text) of 36 seeded .eval files with Python's standard zip/json readers in a Modal container.

## What we found

First user turn empty in 36/36; one non-empty system prompt in 23/36, the rest one-message failed runs (RESULT.md).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-15_q3_interleaved_run3_sim2](../2026-10-15_q3_interleaved_run3_sim2/)

## Main scripts

- [`run_raw_eval_read.py`](run_raw_eval_read.py)

Result files the index names:

- `raw_eval_read_2026-10-15_015821.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_raw_eval_read_09/` at commit `b12f788942ebed5b440f576c45413cf40ccf55c2` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
