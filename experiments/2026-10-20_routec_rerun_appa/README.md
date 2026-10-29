# 2026-10-20_routec_rerun_appa

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Can an independent implementer re-derive the post's Appendix A shares of scored commands that are file edits (about 17% in the honest pool and about 21% in attack runs)?

## What we did

A fresh implementer re-derived the shares from the prepared transcript files under a plan written down before any value was seen, including sensitivity variants an independent reviewer required, then ran a diagnostic of why the honest-pool counts differed.

## What we found

Attack-run share (row 251) reproduced; honest-pool share (row 250) unresolved under the pre-stated rule (PRESTATED_DIAG text).

## In the post?

Not in the post (verifies Appendix A's file-edit shares).

## Status

Current

## Builds on

[2026-10-20_routec_fileedit_share](../2026-10-20_routec_fileedit_share/)

## Main scripts

- [`run_appa.py`](run_appa.py)
- [`diag_appa.py`](diag_appa.py)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_rerun_appa/` at commit `c470893b54e7eb9a159920ed841344d88c2c25f6` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
