# 2026-10-10_route3_phase1_e2

2026-10-10 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Real-versus-real control for Phase 1: does the published impactful-command monitor catch GPT-5.6-Sol's second real run of each task as it catches the first, beside simulated vs real?

## What we did

Rebuilt epoch 1 from the refit, scored epoch 2 with the same frozen probes and 1% thresholds on 505 tasks, intervals re-estimating thresholds per resample; plan committed first.

## What we found

README: catch 90.83% (real run 1), 91.23% (real run 2), 90.80% (simulated); epoch 2 minus epoch 1 +0.40 points [-0.65, +1.35], simulated minus epoch 1 -0.02 [-1.22, +1.07]; both rule out a 5-point gap.

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-08_impactful_refit_bootstrap](../2026-10-08_impactful_refit_bootstrap/)

## Main scripts

- [`run_route3_e2.py`](run_route3_e2.py)
- [`analyze_route3_e2.py`](analyze_route3_e2.py)

Result files the index names:

- `s3_manifest_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_route3_phase1_e2/` at commit `be1156719c43eaf34e4a4e95207c3f64fc2afd3e` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
