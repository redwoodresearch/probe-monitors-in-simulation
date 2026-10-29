# 2026-10-11_claimB_h1_text_vs_layout

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Hypothesis 1 of the fixed-simulator test: does the realism probe follow what the simulated tool outputs say or how they look, tested by swapping whole outputs between real and simulated runs at identical calls?

## What we did

Counted identical calls (same function and arguments) between simulated and real runs and measured the main realism probe's per-output gap there on stored scores, then ran a paired-swap pilot on the five development environments (Qwen3-32B reads on Modal), with gates on projected uncertainty before any full swap.

## What we found

PLAN.md: at identical bash calls the probe's simulated-minus-real gap is +0.557 [0.406, 0.697] logits vs +0.053 for two real runs (no shuffle null). Pilot projected uncertainty 1.41 against a 0.12 bar, so the full run did not proceed. A gate failed on 102 matched pairs and the swap (about \$12k) was dropped.

## In the post?

Not in the post.

## Status

Exploratory (closed as too few)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (its first probe step)

## Main scripts

- [`analyze_identical_calls.py`](analyze_identical_calls.py)
- [`analyze_identical_call_gap.py`](analyze_identical_call_gap.py)
- [`analyze_POSTHOC_gap_null.py`](analyze_POSTHOC_gap_null.py)
- [`analyze_dev_gate.py`](analyze_dev_gate.py)
- [`run_h1_swap_modal.py`](run_h1_swap_modal.py)
- [`run_h1_test_swap_modal.py`](run_h1_test_swap_modal.py)

Result files the index names:

- `identical_calls_2026-10-11_102728.json` (not in this folder at the source commit)
- `identical_call_gap_2026-10-11_103052.json` (not in this folder at the source commit)
- `PLAN.md` (not copied: notes or README (internal working labels))

## Files here

6 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_h1_text_vs_layout/` at commit `057c4215eaa6a3df4407960b41cdf07b65b06f0f` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
