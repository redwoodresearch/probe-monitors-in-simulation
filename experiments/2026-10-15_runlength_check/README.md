# 2026-10-15_runlength_check

2026-10-15 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do longer runs (more chances to alarm) drive the simulated-vs-real catch gaps or run 2's lower catch on the original set?

## What we did

Capped or matched the number of scored units per run (30, paired, task minimum) and recomputed catches; later sections of its written rules specify the run-level honest alarm rates and the pooled run-level metrics (AUROC) that the final pipeline computed.

## What we found

Run-length variants written for both readouts; values not read for this index. Sim attack runs had more units than real run 1 (+1.6 commands on the +311 added tasks) per its written rules.

## In the post?

Not in the post (its pooled-metrics design underlies Table 1's AUROC column, whose record is in [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/)).

## Status

Exploratory

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/)

## Main scripts

- [`gate_v2.py`](gate_v2.py) (runs in [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) code)

Result files the index names:

- `runlen_v2_C_cap30_2026-10-15_221201.json` (not copied: result file 1 MB or larger)
- `runlen_v2_R_cappaired_2026-10-15_221216.json` (not copied: result file 1 MB or larger)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_runlength_check/` at commit `106262746bbdcd228d5ff84598ef3fc34d0429f9` of our internal repository. 16 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
