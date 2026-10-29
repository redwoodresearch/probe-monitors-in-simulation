# 2026-10-11_claimB_layer0_control

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does a probe on the very first layer (raw token embeddings, before any processing) separate formatting-fixed simulated runs from real ones nearly as well as the main realism probe (0.82 AUROC), which would mean the probe reads words rather than processed meaning?

## What we did

Packed Qwen3-32B's layer-0 token embeddings and first-block outputs for the stored formatting-fixed simulated and real runs on the 5 test environments, refit the main probe's layer and regularization grid on them on Modal with negative and planted controls, and ran a Hewitt-and-Liang control task (can these layers encode the readout token's identity?).

## What we found

RESULT.md on a side branch (f14ae8b93): the main Qwen3-32B end-of-output realism probe 0.822 vs embeddings 0.579, difference 0.243 [0.213, 0.275], but the verdict is 'inconclusive (edge-limited)' because the embedding probe picked the smallest regularization setting in 5 of 10 folds; block 0 0.587; the control task shows these layers can encode the readout token's identity.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (its first probe step); [2026-10-10_claimB_probe](../2026-10-10_claimB_probe/) (grid code)

## Main scripts

- [`power_layer0.py`](power_layer0.py)
- [`check_plant_readout_tokens.py`](check_plant_readout_tokens.py)
- [`run_layer0_pack.py`](run_layer0_pack.py)
- [`run_layer0_grid.py`](run_layer0_grid.py)
- [`analyze_layer0.py`](analyze_layer0.py)
- [`run_control_task_modal.py`](run_control_task_modal.py)

Result files the index names:

- `layer0_analysis_2026-10-11_230533.json` (not in this folder at the source commit)
- `control_task_2026-10-11_231216.json` (not in this folder at the source commit)
- `layer0_grid_E_2026-10-11_230437.json` (not in this folder at the source commit)
- `layer0_grid_B0_2026-10-11_230454.json` (not in this folder at the source commit)
- `power_layer0_2026-10-11_195623.json` (not copied: result left out to keep the release under 120 MB)
- `plant_readout_tokens_2026-10-11_195412.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_layer0_control/` at commit `406002b04b42d11d0b2ac827813a48cb2ba2d3cb` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 4 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
