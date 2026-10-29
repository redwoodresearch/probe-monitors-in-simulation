# 2026-10-08_q2_surface_edit_sweep

2026-10-08 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

If one simulator tell is reversed in real runs (strip the line-number gutter, add a trailing newline, remove the review note), how much does the frozen tool-output attack detector's catch change?

## What we did

Staged edited copies of real GPT-5.6-Sol runs, one tell at a time, with a change report; scored by the Phase 1 frozen tool-output detector (scores in [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)).

## What we found

Catch at 5% fell -7.7 [-12.4, -3.6] (gutter stripped), -8.5 [-12.1, -5.4] (newline added) and -3.8 [-8.1, +0.3] (review note removed) (preliminary).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-08_q2_surface_edits](../2026-10-08_q2_surface_edits/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`run_stage_edits.py`](run_stage_edits.py)
- [`run_change_report.py`](run_change_report.py)

Result files the index names:

- [`stage_report_2026-10-08_144623.json`](results/stage_report_2026-10-08_144623.json)
- `manifest_change_report_v1.json` (not copied: JSON outside results/ and config/)
- `surface_edits_2026-10-08_171856.json` (not in this folder at the source commit)

## Files here

3 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-08_q2_surface_edit_sweep/` at commit `5a335d60f3454ade342b5688cfd15e171d3d860e` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
