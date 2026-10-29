# 2026-10-13_q3_sabotage_separability

2026-10-13 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Are simulated sabotage commands more distinguishable from real sabotage commands than ordinary commands are (is it 'how the attack looks')?

## What we did

Qwen3.5-27B layer 30, own runs: a realism probe (5 main-task folds) on 313 simulated vs 366 real sabotage calls, compared with count-matched ordinary calls, a second-real-run control and a label-shuffle null, at three command readouts. Modal, \$0.55.

## What we found

README (sab_separability_2026-10-13_233027.json): at the primary command last-token readout, sabotage simulated vs real AUROC 0.465 vs 0.486 for ordinary calls (difference -0.021 [-0.081, +0.043]), null p = 0.96, reading 'no'; the averaged command readout separates above null (0.567, p = 0.010) but its edge over ordinary calls is inconclusive.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sabotage_t2_realism](../2026-10-13_q3_sabotage_t2_realism/) (realism-test code); [2026-10-13_q3_sabotage_scaleup](../2026-10-13_q3_sabotage_scaleup/) (labels)

## Main scripts

- [`run_sab_separability.py`](run_sab_separability.py)
- [`t2_core.py`](t2_core.py)
- [`extract_pooled.py`](extract_pooled.py)

Result files the index names:

- `manifest_sabsep_s3_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_sabotage_separability/` at commit `3d35cbd61bc6be3cab526358c91bd39ab61b868e` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
