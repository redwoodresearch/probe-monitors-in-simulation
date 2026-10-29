# 2026-10-10_claimB_qwen35_reader

2026-10-10 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Fixed-simulator test follow-up: does Qwen3.5-27B, as a second reader, also tell the fixed-simulator GPT-5.6-Sol runs from real ones?

## What we did

Packed and scored Qwen3.5-27B activations on the fixed off-policy data with times and dates masked on five test environments, with real-versus-real control gate, label-shuffle nulls and planted checks, pre-registered.

## What we found

README: partly, 0.672 [0.641, 0.702] vs 0.822 for Qwen3-32B, catching 5.9% at a 1% alarm target; real versus real at chance (0.498); both planted checks fail (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary; the plan row is superseded by this run)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-10_claimB_probe](../2026-10-10_claimB_probe/)

## Main scripts

- [`run_reader_pack.py`](run_reader_pack.py)
- [`run_reader_scores_modal.py`](run_reader_scores_modal.py)
- [`run_claimB_grid.py`](run_claimB_grid.py)
- [`run_grid_run_scores_modal.py`](run_grid_run_scores_modal.py)

Result files the index names:

- `claimB_b_grid_2026-10-10_221638.json` (not in this folder at the source commit)
- `results_manifest_v1.json` (not copied: JSON outside results/ and config/)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_claimB_qwen35_reader/` at commit `7f5f4400a00f89059649a7ea3169126a298a95bc` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
