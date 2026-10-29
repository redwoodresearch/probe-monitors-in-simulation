# 2026-10-15_q2_monitor_su311

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Does a transcript-reading monitor (the simulated control evaluations paper's trajectory monitor) catch simulated attacks as often as real ones on the probes' own Qwen3.5 runs?

## What we did

Scored the 196-task cohort and the +311 added tasks with GPT-5.4-nano and GPT-5.4-mini trajectory monitors (3 draws per run, cached) on Modal CPU containers; drift and sameness gates; about \$82 API.

## What we found

196 tasks at 5%: mini, simulated minus mean real -1.9 points [-7.4, +3.2]; nano -0.3 [-6.6, +5.9] (README table).

## In the post?

Not in the post.

## Status

Exploratory (transcript-monitor baseline not in the post)

## Builds on

[2026-10-12_q3_nano_realsim](../2026-10-12_q3_nano_realsim/) (outside range)

## Main scripts

- [`run_monitor_modal.py`](run_monitor_modal.py)
- [`analyze_monitor_modal.py`](analyze_monitor_modal.py)
- [`analyze_breakdown_modal.py`](analyze_breakdown_modal.py)
- [`analyze_nm360_modal.py`](analyze_nm360_modal.py)

Result files the index names:

- `analysis_2026-10-15_022511.json` (not copied: result left out to keep the release under 120 MB)
- `breakdown_POSTHOC_2026-10-15_030445.json` (not copied: result left out to keep the release under 120 MB)

## Files here

10 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-15_q2_monitor_su311/` at commit `1fae2a8553fea68dc2db439d2b756517fb2fb8c9` of our internal repository. 30 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
