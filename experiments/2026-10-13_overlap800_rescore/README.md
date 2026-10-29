# 2026-10-13_overlap800_rescore

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

196 of the 939 simulated attack runs were built from a real honest reference run that is also in the honest set; does dropping those cells (a possible copy channel from real into simulated) move any headline gap?

## What we did

Recomputed Seed-OSS-36B's gap (from the new-reader test) and the simulated-control-evaluation probe grid's headline cells (Qwen3-32B, layers 48 and 63, last-token and averaged readouts) on the 743 cells whose simulations used a fresh honest rollout, from stored scores with the original thresholds and bootstrap, gated on reproducing the full-939 values. \$0.

## What we found

g3_o800_2026-10-13_171247.json: Seed-OSS-36B gap y is 13.09 [9.68, 16.51] on the 743 cells against 13.63 [10.76, 16.87] on all 939 (change -0.54 points; cells built from the original 200 real-honest reference runs (which overlap the compared cohort) alone 15.69). SCE grid results in sce_lt_o800 and sce_avg_o800 (not read in detail).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/); [2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/)

## Main scripts

- [`rescore_g3_o800.py`](rescore_g3_o800.py)
- [`rescore_sce_o800.py`](rescore_sce_o800.py)

Result files the index names:

- [`g3_o800_2026-10-13_171247.json`](results/g3_o800_2026-10-13_171247.json)
- [`sce_lt_o800_2026-10-13_171652.json`](results/sce_lt_o800_2026-10-13_171652.json)
- [`sce_avg_o800_2026-10-13_171652.json`](results/sce_avg_o800_2026-10-13_171652.json)

## Files here

2 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-13_overlap800_rescore/` at commit `ffb76daea2927b091cf7042dd9dac90a73e8e9c8` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
