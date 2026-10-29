# 2026-10-14_q3_approach_explicit

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Make sure every attack launch gives each cell exactly its recorded attack approach, since the September builder silently drops it when the reference carries none.

## What we did

Added an approach guard to the attack-mode launcher (hash and length must match the cell's record; an empty approach accepted only when the registry has no strategy) and tests run on Modal.

## What we found

Fix and tests committed; the experiment's written rules note that on the 1,000 September simulated cells the old fallback would have lost 489 approaches and changed 3.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/)

## Main scripts

- [`test_approach_explicit.py`](test_approach_explicit.py)
- [`run_tests_modal.py`](run_tests_modal.py)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_q3_approach_explicit/` at commit `702e7f9233d7d4871cd62344087de4eae4169e1e` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
