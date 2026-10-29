# 2026-10-15_q3_ungrounded_sim

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Does the simulated attack arm still track the real one when the simulator gets no real honest run of the same task as a reference?

## What we did

Built and launched an ungrounded simulator arm (September simulator, no reference run in its prompt) on about 148 pilot cells; later scored by several probes in other folders.

## What we found

Launch done; simulator \$1,320.91 actual over 148 attempts (spend row 2026-10-15 06:48); probe scoring runs logged in later spend rows; no result read here.

## In the post?

Not in the post (the post lists tasks without a reference run as not covered).

## Status

Exploratory

## Builds on

2026-10-14 pilot honest runs

## Main scripts

- [`build_ug.sh`](build_ug.sh)
- [`make_tasks_ug.py`](make_tasks_ug.py)
- [`make_reference_index.py`](make_reference_index.py)
- [`check_ug_runs.py`](check_ug_runs.py)

Result files the index names:

- `ug_check_runs_2026-10-15_123600_keys.json` (not copied: result left out to keep the release under 120 MB)
- `ug_frozen_manifest_2026-10-15_005643.json` (not copied: result left out to keep the release under 120 MB)

## Files here

28 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-15_q3_ungrounded_sim/` at commit `d297151649107d3fa51174fca9b81cc2cc8a3b15` of our internal repository. 16 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
