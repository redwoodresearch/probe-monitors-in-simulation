# 2026-10-05_activation_smoke_test

2026-10-05 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Can Qwen3-32B read the LinuxArena transcripts on one H100 GPU, and how long would reading the full dataset take?

## What we did

Qwen3-32B read 10 real attack and 10 simulated attack transcripts (GPT-5.6-Sol runs, one task per environment) on Modal, saving per-segment pooled activations at 8 layers; a token-count script projected the full-dataset cost.

## What we found

README result table: 19 transcripts read, 1 skipped over 32k tokens, median 12.9 s per transcript, peak 67.4 GB, 0 non-finite values, projecting 8.4 H100-hours; 52% of real honest runs exceed 32k tokens (README).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

Nothing earlier in this index.

## Main scripts

- [`run_activation_smoke.py`](run_activation_smoke.py)
- [`run_count_tokens.py`](run_count_tokens.py)

Result files the index names:

- `smoke_2026-10-05_230831.json` (not copied: result left out to keep the release under 120 MB)
- `token_counts_2026-10-05_225845.jsonl` (not copied: result file type not copied (.jsonl))

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-05_activation_smoke_test/` at commit `b9099333567a6699e976fd7ef6fb3b0ac2f87ad3` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
