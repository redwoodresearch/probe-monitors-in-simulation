# 2026-10-06_segment_end_extraction

2026-10-06 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Extract Qwen3-32B activations at the end of each tool call and tool output (the per-segment readout for the Phase 1 real-vs-simulated and impactful-call questions).

## What we did

Modal job on 20 H100s: Qwen3-32B read all 3,933 valid GPT-5.6-Sol LinuxArena transcripts (four cohorts), cut at 32,768 tokens, saving 8 layers at five positions per action (pre_call, call_end, call_marker, output_end, output_marker) plus tool-output token log-probabilities; variant configs for text-cleanup levels, moving averages and v4 splits.

## What we found

README: 3,933 transcripts read in 1.07 h, 1,232 truncated, 0 failed, 491,055 positions kept, no non-finite values, 15.5 H100-hours (about 61 dollars).

## In the post?

Not in the post.

## Status

Current

## Builds on

None (replaces [2026-10-06_activation_extraction](../2026-10-06_activation_extraction/))

## Main scripts

- [`run_extract_segment_end.py`](run_extract_segment_end.py)

Result files the index names:

- `extract_2026-10-06_133536.json` (not copied: result file 1 MB or larger)
- `extract_2026-10-06_151618.json` (not copied: result file 1 MB or larger)
- `extract_2026-10-06_121617_test.json` (not copied: result left out to keep the release under 120 MB)

## Files here

18 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-06_segment_end_extraction/` at commit `a1bf201226faac88b69457b7894e0eb47bfe57c8` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
