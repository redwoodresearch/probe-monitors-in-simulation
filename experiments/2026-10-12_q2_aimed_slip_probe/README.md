# 2026-10-12_q2_aimed_slip_probe

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does Qwen3.5-27B encode the simulator's str_replace formatting slip (its plain diff format and trailing newline) in its activations at the tool outputs where the slip appears?

## What we did

On stored activations (one Modal CPU container), fit leave-one-environment-out logistic probes at layers 16, 32, 48 and 63 to tell slip-restored outputs from their fixed twins at the 814 slip sites of the fixed-simulator test's second planted check (150 pairs), for Qwen3.5-27B and for Qwen3-32B as a positive control, with 200 pair-level label-shuffle refits.

## What we found

Result JSON (first copy at cc1cead9): Qwen3.5-27B site AUROC 0.897 (max over layers) vs a null 95th percentile of 0.542, reading "decodable"; newline-only sites 0.501 (its template drops the newline); Qwen3-32B 0.986.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-10_claimB_qwen35_reader](../2026-10-10_claimB_qwen35_reader/)

## Main scripts

- [`run_aimed_slip_probe.py`](run_aimed_slip_probe.py)

Result files the index names:

- `aimed_slip_2026-10-12_123449.json` (not in this folder at the source commit)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q2_aimed_slip_probe/` at commit `69e8946093dd11484dfe8de1561437f948433131` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
