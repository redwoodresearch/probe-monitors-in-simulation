# 2026-10-16_cleanroom_pooled_repro

2026-10-16 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does a second implementation, written only from the pre-registered rules, reproduce the pooled records behind the post's headline numbers?

## What we did

An implementer who had not read the production reducer's computational logic before the comparison re-derived the main pooled record's point estimates (before resampling), then the final records (main results at 1% and 5%, the sweep of alarm levels, and strict alarm levels), the sensitivity labels, a refit of the detector that separates two batches of real honest runs (Table 7, row 5), the honest-run pool membership rebuilt from registered pins without the production run catalogue, and counts of scored actions and file edits. Finally it ran a clean-room bootstrap of the interval ends.

## What we found

Final main record: all 520 headline rows within the flip rule (182 bitwise identical, max relative difference 1.1e-12), 48 thresholds and 94,536 per-run catches equal (NOTE_cleanroom_routec_2026-10-20); the independent bootstrap reproduction of interval endpoints: the main record's intervals reproduced exactly (opt3/FOLLOWUPS_OPT3.md, as relayed).

## In the post?

Not in the post (verifies Figure 1 and Tables 5, 7 and 8 numbers).

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) main, alarm-level-sweep and strict-level records

## Main scripts

- [`cleanroom_reduce_v2.py`](cleanroom_reduce_v2.py)
- [`tier2_membership_v2.py`](tier2_membership_v2.py)
- [`cleanroom_boot_opt3.py`](cleanroom_boot_opt3.py)
- [`honest_refit_cleanroom.py`](honest_refit_cleanroom.py)
- [`cunits_by_tool.py`](cunits_by_tool.py)

Result files the index names:

- `cleanroom_A_compute_2026-10-20_020712.json` (not copied: result left out to keep the release under 120 MB)
- `cleanroom_A18_compute_2026-10-20_040452.json` (not copied: result file 1 MB or larger)
- `cleanroom_compute_2026-10-17_000330.json` (not copied: result left out to keep the release under 120 MB)

## Files here

69 code and configuration files and 65 result files, copied byte for byte from `experiments/2026-10-16_cleanroom_pooled_repro/` at commit `d19554c44f47d5a3b4da84487134168a8ed639f3` of our internal repository. 174 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
