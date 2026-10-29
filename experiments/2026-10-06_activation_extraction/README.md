# 2026-10-06_activation_extraction

2026-10-06 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Extract Qwen3-32B activations for every LinuxArena transcript under 32k tokens for the first real-vs-simulated probe analysis.

## What we did

Modal job on 8 H100s: Qwen3-32B read 2,701 transcripts (real honest, real attack epochs 1 and 2, simulated attack; GPT-5.6-Sol runs), saving per-segment mean and moving-average-max pooled activations at 8 layers; transcripts over 32k excluded.

## What we found

Setup only: pooled activations for 2,701 transcripts at 8 layers; RUN.md records a resume after a stop with 2,100 of 2,701 already saved.

## In the post?

Not in the post.

## Status

Superseded by [2026-10-06_segment_end_extraction](../2026-10-06_segment_end_extraction/) (kept for a pooled-vs-segment-end comparison)

## Builds on

[2026-10-05_activation_smoke_test](../2026-10-05_activation_smoke_test/) (token counts and time fit)

## Main scripts

- [`run_extract.py`](run_extract.py)

Result files the index names:

- `extract_2026-10-06_123859.json` (not copied: result left out to keep the release under 120 MB)
- `runs_2026-10-06_123859.jsonl` (not copied: result file type not copied (.jsonl))

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-06_activation_extraction/` at commit `8113946dec6f9d2417f804e85d52c3b84522c15d` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
