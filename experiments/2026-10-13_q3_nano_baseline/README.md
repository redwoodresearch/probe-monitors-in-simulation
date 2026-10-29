# 2026-10-13_q3_nano_baseline

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

How useful is the simulation-trained probe recipe compared with the simulated control evaluations paper's trusted GPT-5.4-nano trajectory monitor on the same real runs?

## What we did

On 937 shared real attack runs and 990 real honest runs (GPT-5.6-Sol), applied the same per-fold threshold rule to GPT-5.4-nano's stored scores (mean of 3 draws) and to the recipe's probes (trained on simulated attack vs simulated honest runs with all formatting repairs, token-averaged tool-output readout at layer 48; Qwen3-32B with an extra fix for trailing newlines on file-view outputs, and Seed-OSS-36B), plus a matched-false-alarm amendment. One Modal CPU function.

## What we found

README (nano_baseline_2026-10-13_225847.json): at a nominal 1% false alarm, nano catches 14.7% [11.3, 19.1] of real attacks against 69.6% (Qwen3-32B recipe) and 62.1% (Seed recipe), paired gaps +54.9 and +47.4 points; at matched false alarms (nano's 1.01%) 65.6% and 57.5%. All within predictions.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_q3_nano_realsim](../2026-10-12_q3_nano_realsim/) (nano scores); [2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (recipe scores, sce_negcount files)

## Main scripts

- [`analyze_nano_baseline.py`](analyze_nano_baseline.py)
- [`analyze_nano_matched.py`](analyze_nano_matched.py)
- [`report_nano_baseline.py`](report_nano_baseline.py)
- [`report_nano_matched.py`](report_nano_matched.py)

Result files the index names:

- `nano_baseline_2026-10-13_225847.json` (not copied: result left out to keep the release under 120 MB)
- `nano_matched_2026-10-13_231840.json` (not copied: result left out to keep the release under 120 MB)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_nano_baseline/` at commit `5a02c66744a4035a29080ac05aaccce11c781000` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
