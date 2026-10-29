# 2026-10-15_u_coverage_09

2026-10-15 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

How often does the upper limit on the "error to add" (how many points a lab should add to its error bars when it measures probe catch in simulation) cover the true error, in simulated datasets that look like ours?

## What we did

A simulation study that planted patterns of simulated-vs-real shift into the version-6 pilot scores, generating 13,500 datasets, and measured how often the upper limit covered the truth and how often the estimate was wrongly zero.

## What we found

Coverage 1.000 in most settings; 0.988 for a single +10-point probe shift (ucov_full table).

## In the post?

Not in the post.

## Status

Exploratory (estimator not in the post)

## Builds on

2026-10-14 error-to-add estimate, version 6

## Main scripts

- [`run_dump_v6.py`](run_dump_v6.py)
- [`run_coverage.py`](run_coverage.py)
- [`run_ucal.py`](run_ucal.py)
- [`run_cont_cov.py`](recheck_contcov_005745/run_cont_cov.py)

Result files the index names:

- `ucov_full_2026-10-15_001850.json` (not copied: result file 1 MB or larger)
- `ucal_2026-10-15_004655.json` (not copied: result file 1 MB or larger)
- `contcov_2026-10-15_005745.json` (not copied: result file 1 MB or larger)

## Files here

14 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_u_coverage_09/` at commit `761fbc925e89e8d114058f13d4d48e48c458db37` of our internal repository. 24 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
