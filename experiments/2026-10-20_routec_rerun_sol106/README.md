# 2026-10-20_routec_rerun_sol106

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Independently check the share of simulated attack runs in which GPT-5.6-Sol wrote part of the simulator output (footnote, about 42%).

## What we did

Fresh implementer recounted the fallback share from run headers and event streams under a pre-statement, with diagnostics when header reads failed.

## What we found

Status PASS, PRELIMINARY; disclosed as not blind to the target count (NOTE_F_result and addendum).

## In the post?

Not in the post (verifies the simulator-fallback footnote).

## Status

Current

## Builds on

[2026-10-20_routec_simpool_numbers](../2026-10-20_routec_simpool_numbers/)

## Main scripts

- [`run_sol106.py`](run_sol106.py)
- [`diag_sol106_ids.py`](diag_sol106_ids.py)
- [`diag3_sol106_events.py`](diag3_sol106_events.py)

Result files the index names:

- `sol106_results_2026-10-20_171433.json` (not copied: result file 1 MB or larger)
- `diag_results_2026-10-20_145435.json` (not copied: result file 1 MB or larger)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_rerun_sol106/` at commit `b5b88e60c963c3dafe17b08678b2d1b4de4aee6e` of our internal repository. 43 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
