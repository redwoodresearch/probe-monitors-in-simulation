# 2026-10-14_format_smell_09

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do empty think blocks or other turn-structure features of the rendered transcripts track the attack/honest label, so that probes could read formatting instead of behaviour? If so, what causes it and how much does a GPT-5.6-Sol-trained probe lean on it?

## What we did

Counts-only Modal CPU scans of the stagings of Qwen3.5-27B's own runs and GPT-5.6-Sol's runs (think-block counts, user turns, run length, tool turns per arm), a cause analysis of the late user turn that strips earlier think blocks under Qwen3.5's chat template, and a rescore of the Sol-trained probe with the affected runs dropped.

## What we found

Found a label-correlated rendering cue: in the Qwen3.5-27B-formatted token-ID inputs, 44 of 404 honest runs vs 10 of 678 attack runs had zero think blocks (p 1.7e-11), caused by a genuine user turn late in the run; the nudge-message explanation was not supported (cue_cause printout). Dropping the 36–75 affected Sol runs moved the Sol-trained probe's catch by under 1 point at layers 48 and 63 (sol_lean printout). The finding led to the corrected render used later.

## In the post?

Not in the post.

## Status

Current

## Builds on

None stated

## Main scripts

- [`run_format_smell.py`](run_format_smell.py)
- [`run_cue_cause.py`](run_cue_cause.py)
- [`run_sol_lean.py`](run_sol_lean.py)

Result files the index names:

- [`format_smell_2026-10-14_102229.json`](results/format_smell_2026-10-14_102229.json)
- `cue_cause_2026-10-14_103513.json` (not copied: result left out to keep the release under 120 MB)
- [`sol_lean_2026-10-14_105630.json`](results/sol_lean_2026-10-14_105630.json)

## Files here

3 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-14_format_smell_09/` at commit `87b31cf5736ff19a0fa1eb2a4f58be074e07cf43` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
