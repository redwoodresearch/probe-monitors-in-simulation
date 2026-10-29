# 2026-10-09_q2_threshold_in_sim

2026-10-09 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

If the impactful-command probe's 1% alarm threshold is set on simulated runs, do real false alarms land between 0.5% and 2% (the project lead's second margin)?

## What we did

Set the 1% threshold on simulated read-only calls (each test fold's calibration fold) and applied it unchanged to real calls, off-policy (530 tasks) and on-policy (20 tasks), for a simulation-trained and a real-trained probe; predictions first.

## What we found

Off-policy real false alarms 1.07% [0.55, 1.74], inside the band; on-policy 3.3%. Thread closed, cannot be settled as registered because two real batches differ about as much.

## In the post?

Not in the post.

## Status

Closed

## Builds on

[2026-10-09_q2_tpr_at_1pct](../2026-10-09_q2_tpr_at_1pct/); [2026-10-08_impactful_refit_bootstrap](../2026-10-08_impactful_refit_bootstrap/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`analyze_threshold_in_sim.py`](analyze_threshold_in_sim.py)

Result files the index names:

- `threshold_in_sim_2026-10-10_110358.json` (not in this folder at the source commit)
- `manifest_v2.json` (not copied: strings over 300 characters)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_q2_threshold_in_sim/` at commit `54337c94fa0db022edd103c304a7d672c1b3d3dc` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
