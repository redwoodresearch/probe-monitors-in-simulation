# 2026-10-08_q2_year_swap

2026-10-08 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does changing the year 2025 to 2026 in simulated tool-output dates change the detector's catch?

## What we did

Staged simulated runs with the year swapped for scoring by the frozen tool-output detector (scores in [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)); predictions committed first.

## What we found

Barely; catch rose +1.4 points [0.0, +3.7] on the 35 of 217 runs changed, so the year is not a fix-list item (preliminary).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-08_q2_date_histogram](../2026-10-08_q2_date_histogram/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`run_stage_year.py`](run_stage_year.py)

Result files the index names:

- `surface_edits_2026-10-08_175911.json` (not in this folder at the source commit)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_q2_year_swap/` at commit `f8a8c7ada877fde3b5548ed8eabf72696cd8000d` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
