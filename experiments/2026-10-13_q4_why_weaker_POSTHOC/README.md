# 2026-10-13_q4_why_weaker_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Why is the 'simulated = attack' shortcut of a mixed-trained tool-output probe weaker when Qwen3.5-27B reads its own runs than when probes read GPT-5.6-Sol's runs: reply position, ceiling effects, input format, or the runs themselves?

## What we did

Rescored saved rows and re-extracted Qwen3.5-27B on Sol's runs at early replies, on its own full transcripts, and in one common reader format across a reader x runs 2x2 (staging, Modal extraction, cells with nulls), measuring the share of the room above real catch that the mixed probe uses. Modal GPU.

## What we found

README (Codex-checked; partly pending a cue re-read): Qwen3.5 reading Sol's runs is also weak at the 3rd reply (6.2% and 1.3% of the room) but uses about 89% by the 21st; on its own full transcripts the on-policy probe catches 98% of real attacks by reply 21, a ceiling; in one common format Qwen3.5 on its own runs uses 45.7% and 54.2% of the room, so format does not explain the gap; the remaining gap tracks the run datasets.

## In the post?

Not in the post.

## Status

Exploratory (preliminary; depends on a later cue re-read)

## Builds on

[2026-10-13_rr_2x2_POSTHOC](../2026-10-13_rr_2x2_POSTHOC/); [2026-10-13_why_steps](../2026-10-13_why_steps/) (saved rows and weights); [2026-10-13_q3_qwen3_crossread](../2026-10-13_q3_qwen3_crossread/) (staging)

## Main scripts

- [`run_q4a_sol_positions.py`](run_q4a_sol_positions.py)
- [`run_q4full_onpolicy_positions.py`](run_q4full_onpolicy_positions.py)
- [`stage_q4b.py`](stage_q4b.py)
- [`extract_q4b.py`](extract_q4b.py)
- [`run_q4b_cells.py`](run_q4b_cells.py)
- [`run_q4b_cells_null.py`](run_q4b_cells_null.py)
- [`run_q4b_fit_steps.py`](run_q4b_fit_steps.py)
- [`run_q4c_matched_paired.py`](run_q4c_matched_paired.py)
- [`extract_q4c.py`](extract_q4c.py)
- [`run_q4d_tells.py`](run_q4d_tells.py)
- [`run_q4e_tells_use.py`](run_q4e_tells_use.py)

Result files the index names:

- `q4a_sol_positions_POSTHOC_2026-10-13_231822.json` (not in this folder at the source commit)
- `manifest_q4_v6.json` (not copied: strings over 300 characters)

## Files here

31 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q4_why_weaker_POSTHOC/` at commit `6a66d0f9286a09e5e845e2b7cf7b64eba76e3c57` of our internal repository. 50 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
