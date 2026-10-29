# 2026-10-12_claimB_plant1_layers

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Is planted check 1 (a probe detecting one inserted line) failing because of the layer or the readout: does it pass at early layers or with mean or max pooling over tokens?

## What we did

\$0 sweep over stored score grids of the experiment comparing last-token, mean-pooled and max-pooled output readouts (3 readouts: last content token, mean and max over tokens; 8 layers 19-63), then a post hoc diagnostic re-extraction on Modal at Qwen3-32B layers 1, 2, 4 and 8 with a probe aimed at the line's tokens and a run-label probe.

## What we found

Codex fact-checks: all 24 stored readout-layer settings miss the 0.95 AUROC bar (best last token at layer 63, 0.853); the aimed probe detects the line at layers 2, 4, 8 (1.0, 1.0, 0.973) but not layer 1 (0.885); the run-label probe detects none (best 0.903).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-10_claimB_probe](../2026-10-10_claimB_probe/); [2026-10-09_fixed_sim](../2026-10-09_fixed_sim/)

## Main scripts

- [`analyze_POSTHOC_plant1_layers.py`](analyze_POSTHOC_plant1_layers.py)
- [`run_POSTHOC_diag_modal.py`](run_POSTHOC_diag_modal.py)

Result files the index names:

- `codex_factcheck_plant1_layers_POSTHOC_2026-10-12_115019.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `codex_factcheck_plant1_diag_POSTHOC_2026-10-12_121250.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `results_manifest_v1.json` (not copied: JSON outside results/ and config/)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_plant1_layers/` at commit `94b94f31a7c73cf7ca0bfccb7cd6632d466ae301` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
