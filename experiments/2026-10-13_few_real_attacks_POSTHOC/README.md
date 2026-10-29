# 2026-10-13_few_real_attacks_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

If a lab trains its detector on simulated attacks vs simulated honest runs, how many real attack runs must it add before the detector catches real attacks about as well as a real-trained one?

## What we did

Added k = 10, 30 or 100 first-real-run attack transcripts (from training tasks only, 5 draws each) to the fully fixed simulated/simulated training set of the Qwen3-32B tool-output probe (reading GPT-5.6-Sol runs, layers 48 and 63), and scored the second real run against the real-trained cell, with 2,000 main-task resamples. Two Modal H100 containers, \$0.08.

## What we found

fewreal_metrics_POSTHOC_2026-10-13_171202.printout.txt: at layer 48, 1%, real catch rises only from 20.9% (k = 0) to 22.8% (k = 100) against 36.9% for the real-trained detector; D (second-real-run catch under the tested probe minus catch under the full-count real-trained probe) -16.0 [-22.5, -7.1] at k = 0 and -14.1 [-20.4, -5.3] at k = 100; 'smallest k with D lower end above -10: none up to 100' at both layers and rates.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (another working session's full-fix grid, cell b'')

## Main scripts

- [`run_fewreal.py`](run_fewreal.py)
- [`analyze_fewreal.py`](analyze_fewreal.py)
- [`percell_fewreal.py`](percell_fewreal.py)

Result files the index names:

- `fewreal_metrics_POSTHOC_2026-10-13_171202.printout.txt` (not copied: lines over 300 characters)
- `manifest_fewreal_v1.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_fewreal_v2.json` (not copied: result left out to keep the release under 120 MB)

## Files here

6 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_few_real_attacks_POSTHOC/` at commit `da8558227fb950d027a255aa98e1851f09f36f06` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
