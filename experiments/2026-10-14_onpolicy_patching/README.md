# 2026-10-14_onpolicy_patching

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

If real tool outputs in Qwen3.5-27B's own runs are replaced by simulator-written ones (insertion) or removed (placeholders), how much do the probe scores move, compared with the observed simulated-vs-real gap?

## What we did

Pilot on 10 honest and 10 attack Qwen3.5-27B runs: the September simulator (GPT-6-Astra) rewrote the first 20 tool outputs of each run with the agent's actions kept byte-identical; Qwen3-32B read the original and rewritten versions, and three frozen main probes trained on GPT-5.6-Sol's runs (mixed, simulation-only and real-trained) scored them at tool-output and command readouts, layers 48 and 63; post hoc checks of the prompt-prefix gate, reasoning blocks and one late-message run.

## What we found

Descriptive pilot: on honest runs (changed outputs, reply layer 48), the probe trained on simulated attacks versus real honest runs rose +0.35 (95% +0.08 to +0.65) and the real-trained probe fell −0.17 (95% −0.35 to −0.01) (patching_scores_2026-10-14_105529.json, summary). Caveats in the file: early-run content only, off-policy reader, 10 + 10 runs; a later caveat says the replayed simulator inputs did not match live runs (RULE.md caveat, 10-15).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_onpolicy_sim](../2026-10-14_onpolicy_sim/)

## Main scripts

- [`run_step0_draw.py`](run_step0_draw.py)
- [`score_patching.py`](score_patching.py)
- [`check_POSTHOC_prefix_and_think.py`](check_POSTHOC_prefix_and_think.py)
- [`check_POSTHOC_late_message.py`](check_POSTHOC_late_message.py)
- [`evalzip.py`](evalzip.py)

Result files the index names:

- [`step0_draw_2026-10-14_094447.json`](results/step0_draw_2026-10-14_094447.json)
- `patching_scores_2026-10-14_105529.json` (not copied: result left out to keep the release under 120 MB)
- [`check_POSTHOC_prefix_and_think_2026-10-14_110353.json`](results/check_POSTHOC_prefix_and_think_2026-10-14_110353.json)
- `check_POSTHOC_late_message_2026-10-14_110807.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

5 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-14_onpolicy_patching/` at commit `397da0f0488cf2609d83ab132a0ec6965d8b25a5` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
