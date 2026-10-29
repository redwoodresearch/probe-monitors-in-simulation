# 2026-10-15_q2_more_runs_variance

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Would more runs or more tasks tighten the interval on the simulated-vs-real catch gap?

## What we did

Decomposed the pooled 455-task interval into task and threshold variance and predicted half-widths for a second simulated epoch, a third real epoch, both, or about 490 more tasks; follow-ups split threshold noise by side and on the full-length record.

## What we found

Joint half-width change: second simulated epoch -5% to -13%; third real epoch -2% to -3%; about 490 more tasks with growing pools -30.6% (README).

## In the post?

Not in the post.

## Status

Exploratory (informed adding the remaining paper tasks)

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) pooled455v2 records

## Main scripts

- [`run_more_runs_variance.py`](run_more_runs_variance.py)
- [`run_threshold_split.py`](run_threshold_split.py)

Result files the index names:

- `morerunsvar_2026-10-15_232948_ap-nTcxHlx47CIrhsh699vCwv.json` (not copied: result left out to keep the release under 120 MB)
- `thrsplit_2026-10-15_235351_ap-7pt643IahPdxaTBcRfjDRj.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_q2_more_runs_variance/` at commit `be3c25c51932d1a85d62f1af8bbcc5f4c5891701` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
