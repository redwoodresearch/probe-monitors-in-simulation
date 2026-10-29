# 2026-10-15_il_z_e2e

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Within one interleaved launch, how far is simulated catch from real catch when each side uses its own honest-run threshold?

## What we did

Computed Z = catch(second simulated run) minus catch(third real run) per action at 1% on the original task set, on the +311 added tasks, and pooled, plus per-set root-mean-square gaps and fold censuses; 12-container Modal CPU passes (about \$8.78 total).

## What we found

Five passes of Z on the +311 added tasks (fifth pass is the reported one); values not read for this index.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-15_q3_interleaved_run3_sim2](../2026-10-15_q3_interleaved_run3_sim2/); [2026-10-15_il_analysis_09](../2026-10-15_il_analysis_09/)

## Main scripts

- [`run_z.py`](run_z.py)
- [`run_znew.py`](run_znew.py)
- [`run_zpool.py`](run_zpool.py)
- [`run_rms_hits.py`](run_rms_hits.py)

Result files the index names:

- `ilz_2026-10-15_125500.json` (not copied: result left out to keep the release under 120 MB)
- `ilznew_2026-10-15_154306.json` (not copied: strings over 300 characters)
- `zpool_2026-10-15_154604.json` (not copied: strings over 300 characters)

## Files here

24 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-15_il_z_e2e/` at commit `f44390c4aa1edd91030c5017f517eb27feb01fbd` of our internal repository. 170 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
