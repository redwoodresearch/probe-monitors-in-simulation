# 2026-10-14_q2_b200_cella_repro

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Were the Qwen3.5-27B activations used to train the frozen off-policy probes in cell (a) of the on-policy figure (layer-63 tool-output probes trained on GPT-5.6-Sol's runs) affected by the numerical mismatch seen on B200 GPUs? The dump requested H200 GPUs but did not record which GPU each shard actually used, so the plan was to re-read 2 runs per shard on a pinned H200.

## What we did

A check-only CPU job verified which shard each transcript belonged to and drew 80 runs (seed 20261014); the GPU re-read on 4 H200 containers was meant to compare their stored activations at all 64 layers.

## What we found

No result: the billing watcher stopped the GPU run at \$5.27 against a \$5 stop (billed \$5.97) before it wrote anything; the check-only run's gates passed (3,933 transcripts, 80 of 80 positions identical). Not relaunched (NOTE_stop_2026-10-14.md).

## In the post?

Not in the post.

## Status

Void

## Builds on

[2026-10-06_qwen35_replication](../2026-10-06_qwen35_replication/) (dump)

## Main scripts

- [`run_cella_repro.py`](run_cella_repro.py)
- [`reader_e08de48.py`](reader_e08de48.py)

Result files the index names:

- [`cella_repro_check_2026-10-14_131215.json`](results/cella_repro_check_2026-10-14_131215.json)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_q2_b200_cella_repro/` at commit `dbca20b695ed94456746d52a2f4a7393c89b94c3` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
