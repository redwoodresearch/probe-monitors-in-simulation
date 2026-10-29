# 2026-10-17_step2_pilot_scoring

2026-10-17 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Score the 20 pilot simulated honest runs with the on-policy probes so they can join the final simulated threshold pool.

## What we did

Launch package and keys-only checks for the on-policy scorer's regression and pilot scoring through the records wrapper; value records parsed only in containers.

## What we found

Regression censuses and keys-only checks written; scoring launched by the coordinator.

## In the post?

In the post: indirect: simulated honest pool for the simulated threshold (Figure 1)

## Status

Current

## Builds on

[2026-10-17_records_step_container](../2026-10-17_records_step_container/); [2026-10-16_own_all_rs](../2026-10-16_own_all_rs/)

## Main scripts

- [`run_step2_regress_ownall.py`](run_step2_regress_ownall.py)
- [`check_b_coverage.py`](check_b_coverage.py)
- [`prove_simh_world_table.py`](prove_simh_world_table.py)

Result files the index names:

- `regress_compare_ownall_census_2026-10-17_123425.json` (not in this folder at the source commit)
- `simh_world_table_proof_2026-10-17_015954.json` (not in this folder at the source commit)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. Other result files, from earlier or side runs, give 7 script-and-commit pairs that match and 2 where the script was not in the folder; we did not copy those versions.

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-17_step2_pilot_scoring/` at commit `0b973ed8815f58e0367495a2f835ae8b42361f29` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 3 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: low). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
