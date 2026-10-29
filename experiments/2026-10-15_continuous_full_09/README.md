# 2026-10-15_continuous_full_09

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Does a score-based (continuous) estimator of the simulated-vs-real catch gap give narrower intervals than counting catches, on the full scale-up cohort?

## What we did

Applied the continuous-score estimator from [2026-10-15_continuous_score_09](../2026-10-15_continuous_score_09/) (byte-identical core) to the all-origins scale-up cohort (per origin and pooled, both readouts, 1% and 5%), after reproducing the overnight gate; Modal CPU, about \$0.2.

## What we found

Records contfull_2026-10-15_121034/121211 written; headline values not read for this index.

## In the post?

Not in the post.

## Status

Exploratory (alternative estimator not used in the post)

## Builds on

[2026-10-15_continuous_score_09](../2026-10-15_continuous_score_09/); [2026-10-14_xerr_scaleup_09](../2026-10-14_xerr_scaleup_09/) (outside range)

## Main scripts

- [`run_cont_full.py`](run_cont_full.py)
- [`run_dump_cont.py`](run_dump_cont.py)
- [`contcore.py`](contcore.py)

Result files the index names:

- [`contfull_2026-10-15_121034.json`](results/contfull_2026-10-15_121034.json)
- `contfull_2026-10-15_121211.json` (not copied: result left out to keep the release under 120 MB)
- `dumpjob_xerr_scaleup_final_2026-10-15_120632.json` (not copied: result file 1 MB or larger)

## Files here

3 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-15_continuous_full_09/` at commit `b6ccaed4650b981c4ae252333257fdbe268e572f` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
