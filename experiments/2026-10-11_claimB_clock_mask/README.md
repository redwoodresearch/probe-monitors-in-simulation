# 2026-10-11_claimB_clock_mask

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does the fixed-simulator-versus-real score of the main Qwen3-32B end-of-output realism probe (ranking score, AUROC, 0.82) move when the clock and date strings left after time masking are also hidden?

## What we did

Masked leftover day-month-year dates, adjoining years and weekday names in tool outputs identically in fixed-simulator, real run 1, real run 2 and planted runs on the 5 test environments, re-read them with Qwen3-32B on Modal and re-scored the main Qwen3-32B end-of-output realism probe (refits), with a planted check.

## What we found

Commit b5a6e236f: 0.82 does not move (fixed simulator vs real run 1 0.8222 to 0.8216, change -0.0006 [-0.0019, +0.0008]); real run 2 vs run 1 0.5365 to 0.5347; planted check passed; catch at 1% 2.7% before and after; the first mask check failed (175 of 176) and was amended; about \$12.2.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (its first probe step); [2026-10-11_claimB_source_audit](../2026-10-11_claimB_source_audit/) (clock-rule leftovers)

## Main scripts

- [`clock_mask.py`](clock_mask.py)
- [`count_clock_mask.py`](count_clock_mask.py)
- [`analyze_power_clock_mask.py`](analyze_power_clock_mask.py)
- [`run_clock_mask_modal.py`](run_clock_mask_modal.py)
- [`analyze_POSTHOC_mask_leftover_modal.py`](analyze_POSTHOC_mask_leftover_modal.py)

Result files the index names:

- `clock_mask_result_2026-10-11_235145.json` (not copied: result left out to keep the release under 120 MB)
- `cells_2026-10-11_230644.json` (not copied: result left out to keep the release under 120 MB)
- `power_clock_mask_2026-10-11_224856.json` (not copied: result left out to keep the release under 120 MB)
- `mask_leftover_POSTHOC_2026-10-11_231554.json` (not copied: result left out to keep the release under 120 MB)
- `mask_leftover_POSTHOC_2026-10-11_231647.json` (not copied: result left out to keep the release under 120 MB)

## Files here

6 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_clock_mask/` at commit `b5a6e236fb033ab3737e6a50d2908e368c09ef09` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
