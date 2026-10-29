# 2026-10-14_il_launch_09

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Which tasks should the interleaved launch (a third real attack run and a second simulated run per task, the analysis spec's numbered addition 32) cover, split into the analysis's estimation and selection halves?

## What we did

A small script built key lists from pinned committed inputs only (no scores read): the estimation and selection sets over the 196 pilot tasks plus the +311 added tasks of the scale-up, and the with-strategy headline cells.

## What we found

Key lists written: 196 + 311 tasks, 97 selection tasks, 99 + 311 estimation tasks (il_keylists_2026-10-14.json, counts).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/) (version 6 error-to-add record); [2026-10-14_q2_scaleup_cohort](../2026-10-14_q2_scaleup_cohort/)

## Main scripts

- [`make_il_keylists.py`](make_il_keylists.py)

Result files the index names:

- `il_keylists_2026-10-14.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_il_launch_09/` at commit `2e75fa8c50978be23f9dc89ee7b04226ef0b4d73` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
