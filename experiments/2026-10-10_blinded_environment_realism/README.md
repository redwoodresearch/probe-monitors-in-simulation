# 2026-10-10_blinded_environment_realism

2026-10-10 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Do the turns that the fake-environment judge flagged actually express the agent's suspicion that its environment is fake, or are they observations of oddities or false positives?

## What we did

Three judges (Gemini 3.1 Pro, Grok 4.7, DeepSeek V4 Pro) via OpenRouter independently rated 124 blinded flagged excerpts from Qwen3.5-27B runs with a fixed rubric, dates and IDs masked, world key never read.

## What we found

Of 124 flagged turns the panel's plurality label was 'suspects fake/test' for 8, 'notices something odd' for 20 and false positive for 94; judge agreement score 0.471 (model judgments, not ground truth).

## In the post?

Not in the post.

## Status

Current (the project lead's own run)

## Builds on

[2026-10-09_phase2_fake_env_judge](../2026-10-09_phase2_fake_env_judge/)

## Main scripts

- [`run_panel.py`](run_panel.py)

Result files the index names:

- `summary.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_blinded_environment_realism/` at commit `70c5197bd568f15444eddb4c3cf0f99e805e20a6` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
