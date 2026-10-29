# 2026-10-10_claimB_small_lm

2026-10-10 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Small-model baseline for the fixed-simulator test: how well does a small language model (Llama-3.2-1B, with a Phi-3.5-mini pilot) tell formatting-fixed simulated runs from real ones through the same probe pipeline?

## What we did

Ran Llama-3.2-1B activations on the formatting-fixed simulated and real GPT-5.6-Sol attack runs through the same probe pipeline, with per-environment analysis and planted checks.

## What we found

Llama scores 0.705 [0.674, 0.735] vs Qwen3-32B's 0.822, near the text classifier; at 1% it catches 7.8% vs 2.7% with wide overlapping intervals; both planted checks miss their bars (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-10_claimB_probe](../2026-10-10_claimB_probe/)

## Main scripts

- [`run_small_lm_pack.py`](run_small_lm_pack.py)
- [`run_small_lm_modal.py`](run_small_lm_modal.py)
- [`analyze_small_lm_per_env.py`](analyze_small_lm_per_env.py)

Result files the index names:

- `claimB_b_grid_2026-10-10_234721.json` (not in this folder at the source commit)
- `results_manifest_v2.json` (not in this folder at the source commit)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_claimB_small_lm/` at commit `6d7caaf5241c32a6faae4e69b804fe7f09bda031` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
