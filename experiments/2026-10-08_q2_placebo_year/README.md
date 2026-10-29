# 2026-10-08_q2_placebo_year

2026-10-08 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Do placebo edits to real runs' tool outputs (a different gutter mark, a same-length neutral sentence instead of the review note) move the tool-output attack detector as much as removing the real tells?

## What we did

Staged placebo-edited copies of real GPT-5.6-Sol runs for extraction and scoring by the frozen Phase 1 tool-output attack detector; scoring results live in [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/).

## What we found

A placebo gutter moved catch -3.2 vs -7.7 for stripping it, but a neutral same-length note moved it -5.7, more than deleting the note (-3.8), so placebos move catch too (preliminary).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-08_q2_surface_edit_sweep](../2026-10-08_q2_surface_edit_sweep/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`run_stage_placebo_year.py`](run_stage_placebo_year.py)

Result files the index names:

- `manifest_extraction_v1.json` (not copied: JSON outside results/ and config/)
- `manifest_inputs_v1.json` (not copied: JSON outside results/ and config/)
- `placebo_pairs_2026-10-08_180653.json` (not in this folder at the source commit)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_q2_placebo_year/` at commit `292cb077602267c93e2f3eacf586f0c71dc2ac7e` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
