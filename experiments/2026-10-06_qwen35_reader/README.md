# 2026-10-06_qwen35_reader

2026-10-06 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Add Qwen3.5-27B as the activation reader (so Phase 1 matches Phase 2, where Qwen3.5-27B is the agent) and count its tokens on every transcript.

## What we did

Reader support in src/probe_shift (pinned Qwen3.5-27B revision, chat-template detection, trimmed messages, XML tool-call segment-end positions) checked on 60 transcripts, a Modal CPU token count of all 3,933 transcripts, and a prepared all-layers extraction script (not run here).

## What we found

README: all 3,933 transcripts parse with 0 failures; Qwen3.5-27B needs 5.4% more tokens than Qwen3; 1,459 transcripts exceed 32k on original text (1,088 after cleanup level 5: time/date and number/ID masking, whitespace and harness-format normalization, and diff-line canonicalization).

## In the post?

Not in the post.

## Status

Current (its Qwen3.5 reader code is likely reused by later Qwen3.5-27B extractions; not verified)

## Builds on

[2026-10-06_qwen35_smoke_test](../2026-10-06_qwen35_smoke_test/)

## Main scripts

- [`run_token_counts.py`](run_token_counts.py)
- [`run_extract_all_layers_qwen35.py`](run_extract_all_layers_qwen35.py)

Result files the index names:

- `qwen35_token_counts_2026-10-06_164545.jsonl` (not copied: result file type not copied (.jsonl))
- [`qwen35_token_counts_2026-10-06_164545.meta.json`](results/qwen35_token_counts_2026-10-06_164545.meta.json)

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-06_qwen35_reader/` at commit `4c692fe0264342afb5d3de1f1cdd95c2be85ff23` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
