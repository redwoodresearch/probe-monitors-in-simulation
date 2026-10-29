# 2026-10-20_routec_reruns

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Coordinate the independent reruns of the final numbers, and independently re-run the planted-error rows (Table 6).

## What we did

The coordinating session's working logs and follow-up lists (10-20 to 10-22, latest release v2.7), and the planted-rows value recheck: the planted-rows producer ported and compared whole-tree with the original plant_rows record.

## What we found

The planted-rows value recheck ran with client exit 0 (plant/NOTE_V3_result); the coordinating session's logs name release v2.7 at 13ca31b07.

## In the post?

Not in the post (verifies Appendix D Table 6; logs track the release).

## Status

Current

## Builds on

[2026-10-15_panel_noise_matched](../2026-10-15_panel_noise_matched/); [2026-10-15_e2e_sim_pipeline](../2026-10-15_e2e_sim_pipeline/)

## Main scripts

- `plant_rows_v3.py` (not in this folder at the source commit) (in [2026-10-15_e2e_sim_pipeline](../2026-10-15_e2e_sim_pipeline/))
- plant RUN_ME launchers

Result files the index names:

- `check_stagea_routec_PLANT_..._foldmodel_2026-10-20_145219.json` (not in this folder at the source commit)

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_reruns/` at commit `edf6430a876df1bfb4bd1ce10eee79e7ca0a36ee` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
