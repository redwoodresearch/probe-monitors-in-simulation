# 2026-10-14_q3_onpol_cue_exclusion

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do the results of the earlier phase's headline probes on Qwen3.5-27B's own runs, and their learning curve, hold when every run carrying a label-correlated cue is excluded? The cue is a genuine user turn, other than the first, before the last tool output, which is far more common in honest runs than in attack runs.

## What we did

Built the list of cue-bearing runs from message roles only, checking that it equals an independently built list, then reran the scoring and the learning-curve readings without the 62 cue-bearing runs, with Qwen3-32B and Seed-OSS-36B reading Qwen3.5-27B's runs.

## What we found

Readings essentially unchanged: Qwen3-32B layer 48 at 5%, repaired-simulation gap +3.7 (95% −1.7 to +10.4) vs +4.1 published; off-to-on drop −41.7 vs −45.4, still "drops" (readings_HELD_2026-10-14.md).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_q3_onpolicy_titlecells](../2026-10-14_q3_onpolicy_titlecells/); [2026-10-14_q3_onpol_learning_curve](../2026-10-14_q3_onpol_learning_curve/); [2026-10-14_format_smell_09](../2026-10-14_format_smell_09/)

## Main scripts

- [`run_cue_step0.py`](run_cue_step0.py)
- [`run_lc_excl.py`](run_lc_excl.py)
- [`score_onpol_excl.py`](score_onpol_excl.py)

Result files the index names:

- `cue_step0_2026-10-14_103103.json` (not copied: result left out to keep the release under 120 MB)
- `onpol_cue_excl_2026-10-14_103450.json` (not copied: result left out to keep the release under 120 MB)
- `readings_HELD_2026-10-14.md` (not copied: lines over 300 characters)

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_q3_onpol_cue_exclusion/` at commit `4d44ad6f4a1263615ed9aaa195fcc3e4e9e3fd77` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
