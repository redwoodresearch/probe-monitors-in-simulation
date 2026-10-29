# 2026-10-09_q2_september_sizing

2026-10-09 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

With the September simulator only, how many tasks does the on-policy comparison need to rule out a real-versus-simulated gap in true positive rate at 1% false positive rate, and what does it cost?

## What we did

Priced simulated runs from the strict rerun's logs and simulated studies by resampling the rerun's 20 tasks (2,000 per size) with fold-consistent refit probes; then wrote the pre-launch plan (PLAN_scaleup.md) for the 196-task September scale-up that the project lead approved.

## What we found

About \$34.31 per added task; 196 tasks give a half-width of about 4.18 points with power 1.00 for a 10-point gap (100 tasks: 5.98, power 0.91-0.93); the project lead then set a 10-point margin on 196 tasks.

## In the post?

Not in the post (its plan launched the 196-task runs that form the post's oldest stratum).

## Status

Current (planning record)

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (strict rerun); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (fold map and refit code)

## Main scripts

- [`price_sim_runs.py`](price_sim_runs.py)
- [`size_september.py`](size_september.py)
- [`size_designs.py`](size_designs.py)
- [`run_refit_fold_probes.py`](run_refit_fold_probes.py)

Result files the index names:

- `sizing_consistent_gap10_2026-10-09_174936.json` (not in this folder at the source commit)
- `manifest_v3.json` (not copied: strings over 300 characters)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_q2_september_sizing/` at commit `e17ca7c5af01632df0e999714c1bccbeaaa50f26` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
