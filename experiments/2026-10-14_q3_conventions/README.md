# 2026-10-14_q3_conventions

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Every headline simulated-vs-real gap thresholded both arms on real honest runs. How do the gaps read when each world is thresholded on its own honest runs (per-world thresholds)?

## What we did

Recomputed published gaps from stored nested scores (earlier phase, GPT-5.6-Sol transcripts) for the mixed "trap" probe on four readers (Qwen3-32B at layers 48 and 63, Qwen3.5-27B, Olmo-3.1-32B, Llama-3.3-70B; not computable for Seed-OSS-36B) and the previously selected main probe configurations on two readers, under shared-real vs per-world thresholds, with task bootstraps; plus a separation analysis.

## What we found

Per-world thresholds reverse the mixed probe's huge shared-real gaps, e.g. Qwen3-32B layer 48 at 5%: +57.6 shared-real vs −13.6 per-world; the previously selected main probe configurations stay near zero (Qwen3-32B all-formatting-repairs, token-averaged tool-output readout, simulated-attack versus simulated-honest training (b), at 5% +0.9 vs −2.1) (conventions_readings_2026-10-14.md).

## In the post?

Not in the post (it motivated the post's per-setting thresholds).

## Status

Exploratory

## Builds on

2026-10-13 sce and why-steps score files

## Main scripts

- [`run_conventions.py`](run_conventions.py)
- [`run_separation.py`](run_separation.py)

Result files the index names:

- `conventions_2026-10-14_060836.json` (not copied: result left out to keep the release under 120 MB)
- `separation_2026-10-14_102455.json` (not copied: result left out to keep the release under 120 MB)
- [`conventions_readings_2026-10-14.md`](results/conventions_readings_2026-10-14.md)

## Files here

3 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_q3_conventions/` at commit `98f3d1a76fb5a201b64c9f698653b569024a73dd` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
