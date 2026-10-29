# 2026-10-14_q3_onpol_fixbyfix

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

On Qwen3.5-27B's own runs, how much of the simulated-vs-real catch gap does each of nine script fixes to the simulator's output formatting close on its own, and how much do all of them close together?

## What we did

Prepared Qwen3.5-27B's simulated runs with each fix alone and with all of them (the same fixes made earlier on GPT-5.6-Sol's runs), extracted activations with the Qwen3-32B and Seed-OSS-36B readers, and scored them with the earlier phase's frozen headline probes trained on real runs, against the first real attack runs and real honest runs.

## What we found

The insert review-note repair, insert trailing-newline repair and clock repair were no-ops on Qwen3.5's runs; as-is gap for Qwen3-32B layer 48, real-trained probe, 1%: +7.8 points (95% +0.4 to +13.0) (fixbyfix_qwen_2026-10-14_114224.json); deltas carry a known same-task leak in the frozen ensemble, leak-free rescore parked (Amendment 5 to the experiment's written rules).

## In the post?

Not in the post.

## Status

Exploratory (leak caveat)

## Builds on

[2026-10-14_q3_onpolicy_titlecells](../2026-10-14_q3_onpolicy_titlecells/); [2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/)

## Main scripts

- [`stage_fixbyfix.py`](stage_fixbyfix.py)
- [`run_extract_qwen3_fx.py`](run_extract_qwen3_fx.py)
- [`run_extract_seed_fx.py`](run_extract_seed_fx.py)
- [`score_fixbyfix.py`](score_fixbyfix.py)
- [`report_fixbyfix.py`](report_fixbyfix.py)
- [`check_fx_qwen.py`](check_fx_qwen.py)

Result files the index names:

- `fixbyfix_qwen_2026-10-14_114224.json` (not copied: result left out to keep the release under 120 MB)
- `fixbyfix_seed_2026-10-14_114240.json` (not copied: result left out to keep the release under 120 MB)
- [`stage_record_fx_2026-10-14_094145.json`](results/stage_record_fx_2026-10-14_094145.json)

## Files here

11 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_q3_onpol_fixbyfix/` at commit `c2b308216b40806c4b2dde21f2705c6d4c1f6aae` of our internal repository. 22 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
