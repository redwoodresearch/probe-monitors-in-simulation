# 2026-10-17_p2_pilot_32k_crosscheck

2026-10-17 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do the published probes' full-length and 32k score files for the 20 pilot simulated honest runs align on command units inside the first 32,768 tokens, and do score-difference diagnostics raise a flag?

## What we did

One Modal CPU check compared unit keys and call indices of the full-length and 32k pilot scores, unit by unit, excluding boundary units under the registered rule. Score differences were diagnostic only; PASS tested alignment, with no score-equivalence tolerance.

## What we found

Alignment PASS: 3,456 units compared, 0 mismatches, keys equal (result JSON); the diagnostic score-difference flag was false. This does not certify score equality.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-17_step2_pilot_scoring](../2026-10-17_step2_pilot_scoring/)

## Main scripts

- [`run_pilot_32k_crosscheck.py`](run_pilot_32k_crosscheck.py)
- [`read_32k_record_keys.sh`](read_32k_record_keys.sh)

Result files the index names:

- `pilot_32k_crosscheck_2026-10-17_190330.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-17_p2_pilot_32k_crosscheck/` at commit `f4d1db66a536fc70ff673892efbdacd66057d7d6` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
