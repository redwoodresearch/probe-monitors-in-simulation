# 2026-10-06_all_layers_extraction

2026-10-06 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Which of Qwen3-32B 64 layers carries the real-vs-simulated signal? (extract every layer at the segment-end positions)

## What we did

Modal extractor that re-reads the staged reader inputs of a segment-end run and saves the residual stream after all 64 Qwen3-32B layers at five positions per action, in per-layer chunk files; run on original text and seven cumulative cleanup stages covering times/dates, numbers/IDs, whitespace, harness formatting, diff lines, empty diff lines and editor-output formatting; the v4 five-arm splits; and many later Phase 1 variant dumps (swaps, planted checks, fixed simulator).

## What we found

README runs table: 3,933 transcripts, 491,055 rows, 0 non-finite values in 0.43 h on 40 H100s (original text); the same extractor later produced the follow-up Qwen3-32B dumps.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-06_segment_end_extraction](../2026-10-06_segment_end_extraction/) (inputs_from staged reader messages)

## Main scripts

- [`run_extract_all_layers.py`](run_extract_all_layers.py)

Result files the index names:

- [`extract_2026-10-06_la_segment-end_qwen3-32b_alllayers_v1_2026-10-06_170701.json`](results/extract_2026-10-06_la_segment-end_qwen3-32b_alllayers_v1_2026-10-06_170701.json)
- [`extract_2026-10-06_la_segment-end-scrub7_qwen3-32b_alllayers_v1_2026-10-06_175640.json`](results/extract_2026-10-06_la_segment-end-scrub7_qwen3-32b_alllayers_v1_2026-10-06_175640.json)
- [`extract_2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1_2026-10-06_214034.json`](results/extract_2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1_2026-10-06_214034.json)

## Files here

47 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-06_all_layers_extraction/` at commit `4aede7d83722605d7279d8414e55341f5d3ce44d` of our internal repository. 20 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
