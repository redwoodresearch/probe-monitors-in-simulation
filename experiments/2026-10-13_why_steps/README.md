# 2026-10-13_why_steps

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Why does the fully repaired mixed probe (trained on simulated attacks vs real honest runs) still overstate catch on simulated runs: is it the reply text, the agent's behaviour, or the simulator's content?

## What we did

With Qwen3-32B (and Seed-OSS-36B) readouts of tool outputs and commands on GPT-5.6-Sol runs, compared the mixed recipe's overstatement at the command readout with that at the tool-output readout, then Step 1 (every tool reply replaced by a placeholder) and Step 2 (30 real runs with simulator-rewritten replies), plus token-averaged readouts and extra formatting fixes (a trailing newline on file-view outputs and two tab-handling fixes); key figures for review. Modal.

## What we found

why_metrics_steps_2026-10-13_175358.json: at layer 48, 1%, the fully repaired mixed probe overstates by +5.6 points [+2.6, +8.6] at the command readout (against about +30 at the tool-output readout) and by +2.2 [-0.9, +5.2] with placeholder replies; reading 'patching and the command readout both needed'. Seed-OSS-36B readings marked SUSPECT (B200 shards not re-read).

## In the post?

Not in the post.

## Status

Exploratory (Seed readings suspect)

## Builds on

[2026-10-13_q2_step1_placeholder](../2026-10-13_q2_step1_placeholder/); [2026-10-13_step2_sim_rewrite](../2026-10-13_step2_sim_rewrite/); [2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/); [2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/)

## Main scripts

- [`run_why_steps.py`](run_why_steps.py)
- [`analyze_why_steps.py`](analyze_why_steps.py)
- [`score_step2.py`](score_step2.py)
- [`plot_steps.py`](plot_steps.py)
- [`plot_key_figure.py`](plot_key_figure.py)
- [`check_weights_rescore.py`](check_weights_rescore.py)

Result files the index names:

- [`why_metrics_steps_2026-10-13_175358.json`](results/why_metrics_steps_2026-10-13_175358.json)
- [`why_metrics_steps_2026-10-13_224058.json`](results/why_metrics_steps_2026-10-13_224058.json)
- `why_ph_run_2026-10-13_175111.json` (not copied: result left out to keep the release under 120 MB)
- [`why_step2_2026-10-13_230153.json`](results/why_step2_2026-10-13_230153.json)
- `key_figure_qwen3-32b_2026-10-13_v5.json` (not copied: strings over 300 characters)
- [`fig_steps_qwen3-32b_2026-10-13_v3.json`](results/fig_steps_qwen3-32b_2026-10-13_v3.json)

## Files here

6 code and configuration files and 5 result files, copied byte for byte from `experiments/2026-10-13_why_steps/` at commit `9edcc585e16e9db391b86a968f2f33def7de8390` of our internal repository. 42 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
