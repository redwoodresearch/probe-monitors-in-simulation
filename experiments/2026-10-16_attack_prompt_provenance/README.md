# 2026-10-16_attack_prompt_provenance

2026-10-16 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Which attack prompt did our attack runs use, is it byte-identical to the public LinuxArena paper prompts, do run limits differ across task sets, and what did the simulator's OpenAI usage cost?

## What we did

Compared prompt file hashes across the sims trees and the public repo (no text printed), checked run limits and models per task set on seeded .eval files, and priced and attributed OpenAI usage from a scan of run logs (Modal containers).

## What we found

OpenAI usage by model: GPT-6-Astra (simulator) \$45,321.68, GPT-5.6-Sol \$1,685.74 (openai_usage_reanalysis_2026-10-16_191817 by_model).

## In the post?

Not in the post (supports the methods description in "What we ran").

## Status

Current

## Builds on

Nothing earlier in this index.

## Main scripts

- [`check_attack_prompt.py`](check_attack_prompt.py)
- [`run_eval_limits_check.py`](run_eval_limits_check.py)
- [`run_openai_usage_scan.py`](run_openai_usage_scan.py)
- [`analyze_openai_usage.py`](analyze_openai_usage.py)

Result files the index names:

- `attack_prompt_provenance_2026-10-16_112230.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `eval_limits_2026-10-16_124417.json` (not copied: result left out to keep the release under 120 MB)
- `openai_usage_reanalysis_2026-10-16_191817.json` (not copied: result file 1 MB or larger)

## Files here

5 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-16_attack_prompt_provenance/` at commit `de0bf0301e47b9d169d753a77f7e5b3fd3e26f0b` of our internal repository. 20 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
