# 2026-10-11_claimB_h3_fix_to_plateau

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

If the simulator's remaining formatting slips are fixed until the gap stops shrinking, is the fixed simulator as close to real runs as a second real run, within a pre-set margin?

## What we did

Power analysis for the smallest testable margin, then a small run that evened the four leftover slips on the test environments, re-read the runs with Qwen3-32B on Modal and scored probe and text-classifier detectors with planted checks.

## What we found

Too few; planted check 1 failed for both detectors (0.794 and 0.829 vs the 0.833 floor), so nothing is read; descriptively, simulated runs stand out beyond real-run noise by 0.268 in ranking score.

## In the post?

Not in the post.

## Status

Exploratory (closed as too few)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-11_claimB_slips_evened_probe](../2026-10-11_claimB_slips_evened_probe/) (the evening rules)

## Main scripts

- [`analyze_power_margin.py`](analyze_power_margin.py)
- [`run_h3_small_modal.py`](run_h3_small_modal.py)
- [`analyze_h3_small.py`](analyze_h3_small.py)
- [`run_POSTHOC_plant1_diag_modal.py`](run_POSTHOC_plant1_diag_modal.py)
- [`run_POSTHOC_plant1_followup_modal.py`](run_POSTHOC_plant1_followup_modal.py)

Result files the index names:

- `h3_small_2026-10-11_155930.json` (not copied: result left out to keep the release under 120 MB)

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_h3_fix_to_plateau/` at commit `808c617401df37fbf1f22005f7eb6290105f1fc7` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
