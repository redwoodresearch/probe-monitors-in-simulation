# 2026-10-20_routec_rerun_rc1b-w1

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Independently re-derive two reported kappa numbers (agreement on which tasks get caught).

## What we did

Fresh implementer read the row store and recomputed kappa groups (commands, joint draws) with toy, perturbation and real launches; Modal CPU, cap \$0.15.

## What we found

Verdict MATCH: 24 of 24 on the real run (W1_NOTE_v11_runs).

## In the post?

Not in the post (verifies Appendix H kappa).

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) kappa record

## Main scripts

- [`W1_check_kappa_slots.py`](W1_check_kappa_slots.py)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_rerun_rc1b-w1/` at commit `2750f6e18487fdeaff151a3fa3e4b1459f840200` of our internal repository. 24 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
