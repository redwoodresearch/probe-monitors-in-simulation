# 2026-10-13_q2_auroc_threshold_free

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Are the tool-output (reply) readers' simulated-vs-real catch gaps confined to the 5% and 1% alarm thresholds, or does a threshold-free comparison (area under the ROC curve) agree?

## What we did

From stored scores, computed the AUROC of attack runs against real honest runs for the second real run, the simulated run and the repaired simulated run, for five reply readers (Qwen3-32B last-token and mean-pooled, Olmo-3.1-32B, Llama-3.3-70B, Qwen3.5-27B; all reading GPT-5.6-Sol runs), plus a within-arm secondary for Qwen3-32B. \$0, local.

## What we found

Second real minus simulated AUROC is +0.036 [+0.027, +0.046] for Qwen3-32B last-token, +0.018 for Olmo-3.1-32B, +0.025 for Llama-3.3-70B (same sign as the catch gaps), within 0.015 of zero for mean-pooled and Qwen3.5-27B; after repair Qwen3-32B and Olmo are within 0.015, Llama keeps +0.010. Within-arm, real 0.934 vs simulated 0.909, difference +0.025 [+0.010, +0.041].

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

The attack detector and the reply-reader runs of 2026-10-12 (q2_llama_reader, q2_olmo_reader, q2_meanpool_36); the repaired arm of all 939 cells

## Main scripts

- [`analyze_auroc.py`](analyze_auroc.py)

Result files the index names:

- `auroc_2026-10-13_011851.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q2_auroc_threshold_free/` at commit `e77d511aae650a2cad7590404398dfe19afc3a55` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
