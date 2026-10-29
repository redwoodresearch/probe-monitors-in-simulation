# 2026-10-13_g3_causal

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

How much of Seed-OSS-36B's per-output simulated-vs-real score shift remains when each tool output is re-read with only its own tool call and no earlier conversation history?

## What we did

Re-read every kept output of the 939 paired task cells (real honest, first and second real attack, simulated attack) behind a fixed minimal context (system prompt, task, the output's own call), scored them with the frozen Seed-OSS-36B detector from the new-reader test ([2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/)), and compared per-output shifts with the in-history shifts over 2,000 main-task resamples. Modal GPU, about \$38 for a cached-context extraction (rejected by its accuracy check) plus a rerun without the cache.

## What we found

README (from ctx_analysis_POSTHOC_2026-10-13_150614.json): over all rows, simulated minus second-real shift is -0.308 logits in history and -0.109 with no history, share kept 0.35 [0.22, 0.48] ('history carries most'); outputs not ending in a newline keep 1.12; str_replace keeps 0.14; insert keeps 0.98.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/) (frozen probes, forecast.json); [2026-10-13_g3_followups](../2026-10-13_g3_followups/) (in-history rows)

## Main scripts

- [`analyze_ctx.py`](analyze_ctx.py)
- [`verify_cache_g3ctx.py`](verify_cache_g3ctx.py)
- [`run_extract_g3ctx_nocache.patch`](run_extract_g3ctx_nocache.patch)
- [`launch/ctx_launch.sh`](launch/ctx_launch.sh)

Result files the index names:

- `ctx_analysis_POSTHOC_2026-10-13_150614.txt` (not copied: lines over 300 characters)
- [`ctx_check_2026-10-13_g3_ctxnc_seed-oss-36b_v1_2026-10-13_144953.json`](results/ctx_check_2026-10-13_g3_ctxnc_seed-oss-36b_v1_2026-10-13_144953.json)
- `s3_manifest_v1.json` (not copied: strings over 300 characters)

## Files here

5 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_g3_causal/` at commit `aa2dbafc77ffad96493c3e7004adbe32050f4369` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
