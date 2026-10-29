# 2026-10-20_routec_numv_recheck

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Independently recheck a seeded sample of the numbers verifier's "equal" rows with separate code.

## What we did

Built a recheck that re-derives 20 seeded rows per verifier pass without importing the verifier or slot code; synthetic tests (27 passed).

## What we found

Run 2 passed: in one batch all 20 rows were equal, in the other 19 were equal and one text row was explicitly left unchecked (NOTE_recheck_independent_2026-10-20.md, lines 311–330); the NOTE's opening "built, not run" status is stale.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-20_routec_numbers_verify](../2026-10-20_routec_numbers_verify/)

## Main scripts

- [`run_recheck_independent.py`](run_recheck_independent.py)
- [`launch_recheck.zsh`](launch_recheck.zsh)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_numv_recheck/` at commit `da0a4960855c5b44bf7ee9406ff1bcf30411bc92` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
