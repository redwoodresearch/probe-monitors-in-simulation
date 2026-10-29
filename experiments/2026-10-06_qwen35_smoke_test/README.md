# 2026-10-06_qwen35_smoke_test

2026-10-06 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Can Qwen3.5-27B, loaded text-only, read the LinuxArena transcripts in place of Qwen3-32B?

## What we did

One H100 on Modal: Qwen3.5-27B read the same 20 transcripts as the Qwen3-32B smoke test whole (up to about 83k tokens), checking memory, speed, hooks, non-finite values and chat-template differences.

## What we found

README: reads all 20 transcripts whole on one GPU (83,131 tokens at 66.1 GB peak), 0 non-finite values; its template trims messages, erasing the trailing-newline tell (52% real vs 95% simulated outputs end in a newline).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-05_activation_smoke_test](../2026-10-05_activation_smoke_test/) (same 20 transcripts)

## Main scripts

- [`run_qwen35_smoke.py`](run_qwen35_smoke.py)

Result files the index names:

- `qwen35_smoke_2026-10-06_123712.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-06_qwen35_smoke_test/` at commit `1fea5f9e85b2d82615fe54c30e2b6049c55ebf5c` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
