# 2026-10-14_eoc_recheck_09

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do the two end-of-call ("any command") sensitivity rows of the preliminary error-to-add analysis (version 6, Amendments 9 and 9b) reproduce with independent code?

## What we did

One Modal CPU job recomputed, from the pinned score inputs and written from the rule text rather than run_xerr_v6.py, the command-readout rows at 1% and 5% for the reselected and frozen-eligibility records, plus the as-run primary, and compared them by key.

## What we found

Both runs reproduced: 613 comparisons, 0 mismatches (eoc_recheck_2026-10-14_192341.json and eoc_recheck_2026-10-14_192600.json, keys result.n_comparisons, result.n_mismatches).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/)

## Main scripts

- [`run_eoc_recheck.py`](run_eoc_recheck.py)

Result files the index names:

- [`eoc_recheck_2026-10-14_192341.json`](results/eoc_recheck_2026-10-14_192341.json)
- [`eoc_recheck_2026-10-14_192600.json`](results/eoc_recheck_2026-10-14_192600.json)

## Files here

1 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-14_eoc_recheck_09/` at commit `25698912207e548e9ae80ba7c92ae6618faee096` of our internal repository. Every file in that folder was copied.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
