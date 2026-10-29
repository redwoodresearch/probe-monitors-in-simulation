# 2026-10-16_fl_vs_32k_appendix

2026-10-16 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

An appendix comparing results within the first 32,768 tokens with full-length results.

## What we did

Built a render plan, training-length sheets and table/plot scripts for a 32k vs full-length appendix record.

## What we found

Only training_lengths records were committed; the final table run's after-run field reads TO_FILL.

## In the post?

Not in the post (the post does not mention the 32k cut).

## Status

Partly completed: training-length records were completed and committed; the appendix was dropped; completion of the final table run is not established by the available records

## Builds on

[2026-10-15_cut_sensitivity](../2026-10-15_cut_sensitivity/)

## Main scripts

- [`run_table_modal.py`](run_table_modal.py)
- [`make_fl_vs_32k_table.py`](make_fl_vs_32k_table.py)
- [`resolve_training_lengths.py`](resolve_training_lengths.py)

Result files the index names:

- `training_lengths_2026-10-16_113641.json` (not copied: result left out to keep the release under 120 MB)

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_fl_vs_32k_appendix/` at commit `bdd447de7feb3d9cdd36fc2f4ea40974c026877a` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
