# 2026-10-16_scorer_consistency

2026-10-16 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Were all probe scores feeding one pool made with the same scorer checkpoint (a release gate for the pooled headline)?

## What we did

Compared scorer weights, standardization and per-row scores across score folders for every lead probe, with a dtype-derived tolerance and negative controls; a pre-built rescore plan if it failed.

## What we found

gate_pass true (scorer_consistency_2026-10-16_225449.json).

## In the post?

Not in the post (gate before the pooled headline's release).

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) inputs

## Main scripts

- [`run_scorer_consistency.py`](run_scorer_consistency.py)
- [`run_b.sh`](run_b.sh)

Result files the index names:

- `scorer_consistency_2026-10-16_225449.json` (not copied: result file 1 MB or larger)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_scorer_consistency/` at commit `ee2392c347c905caf04e1aced14edf9a2352af3d` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
