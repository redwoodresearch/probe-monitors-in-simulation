# 2026-10-08_impactful_refit_bootstrap

2026-10-08 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Is the off-policy impactful-command 2x2 result robust when the bootstrap refits the probe (layer, regularization and calibration re-chosen in every resample)?

## What we did

500 main-task resamples that refit the probe over 16 candidate layers for both readers (Qwen3.5-27B headline, Qwen3-32B secondary) on matched calls, with a 20-draw label-shuffle null; Modal T4, about \$60 approved.

## What we found

All 32 real-versus-simulated comparison intervals include zero; for Qwen3.5-27B the ranking gap is -0.0011 [-0.0041, +0.0016] and the fixed-threshold catch gap -0.1 [-1.3, +1.1] points (preliminary; call counts corrected in the log).

## In the post?

Not in the post.

## Status

Current (off-policy baseline for the on-policy comparison)

## Builds on

[2026-10-08_impactful_call_2x2](../2026-10-08_impactful_call_2x2/); [2026-10-06_qwen35_replication](../2026-10-06_qwen35_replication/)

## Main scripts

- [`run_refit_bootstrap.py`](run_refit_bootstrap.py)
- [`analyze_refit_pauc.py`](analyze_refit_pauc.py)
- [`refit_table.py`](refit_table.py)

Result files the index names:

- [`refit_table_2026-10-08_232032.txt`](results/refit_table_2026-10-08_232032.txt)
- `manifest_v1.json` (not copied: strings over 300 characters)

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-08_impactful_refit_bootstrap/` at commit `e1250de7943b48c71731bdf1b64982ec3c20baba` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
