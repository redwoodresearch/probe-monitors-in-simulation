# 2026-10-06_apollo_qwen_gap

2026-10-06 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Why does Apollo deception probe score lower on Roleplaying when retrained on Qwen3-32B than on Llama-3.3-70B: readout, chat template, off-policy text, layer choice, or model size?

## What we did

Every-layer sweep of Apollo recipe on five models (Llama-3.3-70B, Llama-3.1-8B, Qwen3-32B, Qwen3-14B, Qwen3-8B) with two readouts, plus an on-policy test where Qwen3-32B wrote its own Roleplaying answers graded by GPT-4o; label-shuffle null and held-out layer selection.

## What we found

README: Llama-3.3-70B 0.963 vs Qwen3-32B best layer 0.871 (0.842 with layer picked on held-out data); readout, template and off-policy text ruled out, layer choice explains part; Qwen3-14B 0.894 and Qwen3-8B 0.947 suggest a size trend.

## In the post?

Not in the post.

## Status

Current (the post uses the published FIBS deception probe instead)

## Builds on

[2026-10-05_apollo_deception_probe](../2026-10-05_apollo_deception_probe/); [2026-10-06_apollo_llama_check](../2026-10-06_apollo_llama_check/)

## Main scripts

- [`run_qwen_gap.py`](run_qwen_gap.py)
- [`run_grade_roleplaying.py`](run_grade_roleplaying.py)
- [`analyze_layer_sweep.py`](analyze_layer_sweep.py)
- [`analyze_rerun.py`](analyze_rerun.py)
- [`plot_layer_sweep.py`](plot_layer_sweep.py)

Result files the index names:

- [`layer_sweep_summary_2026-10-07_104042.json`](results/layer_sweep_summary_2026-10-07_104042.json)
- [`roleplaying_by_layer_2026-10-07_104042.csv`](results/roleplaying_by_layer_2026-10-07_104042.csv)
- `sweep_qwen3-32b_2026-10-06_140859.json` (not copied: result file 1 MB or larger)
- `sweep_qwen3-14b_2026-10-07_103957.json` (not copied: result left out to keep the release under 120 MB)
- [`rerun_check_qwen3-8b_2026-10-07_103711.json`](results/rerun_check_qwen3-8b_2026-10-07_103711.json)

## Files here

12 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-06_apollo_qwen_gap/` at commit `a1ee0af08108ec5c9efa7ad022fa9487ff250ce8` of our internal repository. 32 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
