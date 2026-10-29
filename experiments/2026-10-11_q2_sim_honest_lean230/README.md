# 2026-10-11_q2_sim_honest_lean230

2026-10-11 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Buy 230 new GPT-6-Astra simulated honest runs (the September 437-task simulated-honest recipe) to tighten the pooled test of whether monitors catch slightly less on simulated runs than on real ones.

## What we did

Launched stage 1 (50 tasks) on the sims-astra remote development machine with the 437-task recipe's launcher and config pinned by hash and per-task references checked; archived with a header-only sealed manifest.

## What we found

Commit fd92352ca: stage 1 finished 50 of 50, 0 failed, \$301.66; purchase ended after stage 1 and the runs were sealed unread.

## In the post?

Not in the post.

## Status

Stage 1 completed (50 runs, sealed and never read); the continuation was abandoned

## Builds on

[2026-10-11_q2_lean_newruns](../2026-10-11_q2_lean_newruns/); [2026-10-06_astra_sim_honest](../2026-10-06_astra_sim_honest/)

## Main scripts

- [`run_lean230_stage.py`](run_lean230_stage.py)
- [`make_stage_refs.py`](make_stage_refs.py)
- [`build_sealed_manifest.py`](build_sealed_manifest.py)

Result files the index names:

- `lean230_stage1_manifest_sealed.json` (not copied: strings over 300 characters)
- `lean230_stage1_run_ids.json` (not copied: result left out to keep the release under 120 MB)
- `lean230_refs_stage1_2026-10-11_164628.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_q2_sim_honest_lean230/` at commit `fd92352ca94461272fd0af60a1b92045ac72d237` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
