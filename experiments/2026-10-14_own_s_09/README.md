# 2026-10-14_own_s_09

2026-10-14 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Training-domain test: does a probe trained on Qwen3.5-27B's own simulated runs (simulated attack vs simulated honest, plus a variant trained on simulated attack vs real honest) show a smaller simulated-minus-real catch gap than the same recipe trained on real runs?

## What we did

A pre-registered two-phase analysis: phase 1 fixed thresholds and layer sets on 93 selection tasks using only real and honest scores, with simulated scores kept sealed; phase 2 opened them and computed each probe's gap and the difference between the gaps on 90 estimation tasks with 10,000 bootstrap draws stratified by environment, and a later report added a 200-permutation label-shuffle reference for whether the simulation-trained probes separate their classes. The latest ref holds only the pipeline test; the phase records are on a side branch.

## What we found

Phase 2 (own_s_phase2_2026-10-14_233754.json on a side branch), primary cell reply 5%: sim-trained gap −1.7 points (95% −10.0 to +6.5) vs real-trained −12.2 (95% −21.1 to −3.3), difference between the two catch gaps +10.6 (95% +0.5 to +20.7), labelled "Inconclusive at this n". The latest ref's test records are pipeline tests, not results.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_p2_own_sim_trained](../2026-10-14_p2_own_sim_trained/); [2026-10-14_priority1_spec_09](../2026-10-14_priority1_spec_09/); [2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/) (v7 packs)

## Main scripts

- [`run_own_s.py`](run_own_s.py)

Result files the index names:

- `own_s_test_2026-10-14_193345.json` (not copied: result left out to keep the release under 120 MB)
- `own_s_test_2026-10-14_194209.json` (not copied: result left out to keep the release under 120 MB)
- `own_s_phase1_2026-10-14_233434.json` (not in this folder at the source commit)
- `own_s_phase2_2026-10-14_233754.json` (not in this folder at the source commit)
- `own_s_g2report_2026-10-15_002633.json` (not in this folder at the source commit)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_own_s_09/` at commit `5008411e9e316285ae8d5271ed14da0d7d459e24` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
