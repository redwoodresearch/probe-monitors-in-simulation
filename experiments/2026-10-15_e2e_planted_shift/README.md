# 2026-10-15_e2e_planted_shift

2026-10-15 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

Would the end-to-end separate-threshold comparison notice a planted shift in simulated attack scores, and does a shift common to honest and attack runs cancel?

## What we did

On the 183-task pilot cohort, added the smallest constant to simulated attack scores that raises simulated catch by 1, 3, 5 or 8 points (thresholds refit each draw), plus a common-shift negative control; 20 probe rows, Modal CPU.

## What we found

Records plant_2026-10-15_125228/130848 with figures; values not read for this index.

## In the post?

Not in the post.

## Status

Superseded by [2026-10-15_panel_noise_matched](../2026-10-15_panel_noise_matched/) plant_rows on all 1,313 tasks (Table 6)

## Builds on

[2026-10-15_e2e_sim_pipeline](../2026-10-15_e2e_sim_pipeline/) (e2e record 120037)

## Main scripts

- [`run_plant.py`](run_plant.py)
- [`make_captions_v2.py`](make_captions_v2.py)

Result files the index names:

- `plant_2026-10-15_130848.json` (not copied: result left out to keep the release under 120 MB)
- `plant_2026-10-15_130848_plant.png` (not copied: result file type not copied (.png))

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_e2e_planted_shift/` at commit `a82f137247f411cc102de1b660632f0738571510` of our internal repository. 17 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
