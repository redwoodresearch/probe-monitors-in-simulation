# 2026-10-13_q2_avg_vs_last

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does a tool-output readout that averages each output's content tokens (mean-pooled) show the same simulated-vs-real catch gap as the last-content-token readout at the same layers, and does the repaired arm close it under each?

## What we did

From stored nested scores of frozen probes, compared mean-pooled and last-token readouts for Qwen3-32B, Olmo-3.1-32B and Llama-3.3-70B (reading GPT-5.6-Sol runs) on the first and second real runs, the simulated runs and the repaired runs of all 939 cells, with fixed thresholds, planted shifts and a paired contrast on shared main-task draws. \$0, local.

## What we found

paired_readouts_POSTHOC_2026-10-13_115753.txt: at the deep layers the last-token gap is much larger than the mean-pooled one (Qwen3-32B layer 63, 1%: simulated minus second real -10.4 last-token vs -1.1 mean-pooled, contrast +9.3 [+6.0, +12.5]; Olmo layer 56: -11.9 vs -0.9); after repair neither readout shows a difference; at Qwen3-32B layer 48 the readouts do not differ.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_q2_meanpool_36](../2026-10-12_q2_meanpool_36/); [2026-10-12_q2_meanpool_readers](../2026-10-12_q2_meanpool_readers/); [2026-10-12_q2_pinned_layers](../2026-10-12_q2_pinned_layers/); [2026-10-12_q2_olmo_reader](../2026-10-12_q2_olmo_reader/); [2026-10-12_q2_llama_reader](../2026-10-12_q2_llama_reader/)

## Main scripts

- [`analyze_avg_vs_last.py`](analyze_avg_vs_last.py)
- [`analyze_paired_readouts.py`](analyze_paired_readouts.py)

Result files the index names:

- [`avg_vs_last_POSTHOC_2026-10-13_110005.txt`](results/avg_vs_last_POSTHOC_2026-10-13_110005.txt)
- [`paired_readouts_POSTHOC_2026-10-13_115753.json`](results/paired_readouts_POSTHOC_2026-10-13_115753.json)
- [`paired_readouts_POSTHOC_2026-10-13_115753.txt`](results/paired_readouts_POSTHOC_2026-10-13_115753.txt)

## Files here

2 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-13_q2_avg_vs_last/` at commit `3a6a712af490ad14a40cc422db1e95193e7368f7` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
