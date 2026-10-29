# 2026-10-15_s4_orphan_check

2026-10-15 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Did any record relayed on 2026-10-15 use an orphan row from the superseded roots of the fourth new-main-task launch split?

## What we did

Read-only key audit of each record's folder attributions against the source map of the fourth new-main-task launch split.

## What we found

Every audited record "clean" (result JSON summary).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) records

## Main scripts

- [`check_s4_orphans.py`](check_s4_orphans.py)

Result files the index names:

- `s4_orphan_check_2026-10-15_173451.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_s4_orphan_check/` at commit `f5c77c5c0c863cc7e6829bd52b67f0de61dc2af2` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
