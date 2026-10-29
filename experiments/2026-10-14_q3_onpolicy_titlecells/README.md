# 2026-10-14_q3_onpolicy_titlecells

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Do the earlier phase's frozen headline probes (read by Qwen3-32B and Seed-OSS-36B and trained on GPT-5.6-Sol's runs) catch Qwen3.5-27B's own simulated and real attacks equally often, and how much catch do they lose against the same recipe trained on Qwen3.5-27B's own runs?

## What we did

Prepared Qwen3.5-27B's own runs without text scrubbing, plus simulated attacks with all formatting fixes (with and without the file-view trailing-newline repair), extracted activations with both readers, scored them with the frozen ensembles of 20 cross-validated models, and computed whether simulated catch matches real catch with Qwen3.5-27B's own thresholds, the drop from on-policy to off-policy training, and revised intervals.

## What we found

Qwen3-32B layer 48 at 5%: repaired-simulation gap +4.1 points (95% −1.3 to +11.2), "not established" within ±5; off-to-on drop −45.4 (published). Later checks found a task leak and a cue, neither changing the labels (readings_v2_2026-10-14.md; leakfix and cue-exclusion folders).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

2026-10-13 title-cell fits; Qwen3.5-27B pilot runs (196 tasks)

## Main scripts

- [`stage_onpol.py`](stage_onpol.py)
- [`run_extract_qwen3_onpol.py`](run_extract_qwen3_onpol.py)
- [`run_extract_seed_onpol.py`](run_extract_seed_onpol.py)
- [`score_onpol.py`](score_onpol.py)
- [`intervals_v2_onpol.py`](intervals_v2_onpol.py)
- [`make_readings_v2.py`](make_readings_v2.py)

Result files the index names:

- `onpol_titlecells_2026-10-14_024903.json` (not copied: result left out to keep the release under 120 MB)
- `intervals_v2_2026-10-14_031656.json` (not copied: result left out to keep the release under 120 MB)
- [`readings_v2_2026-10-14.md`](results/readings_v2_2026-10-14.md)
- [`stage_result_2026-10-14_011819.json`](results/stage_result_2026-10-14_011819.json)

## Files here

17 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-14_q3_onpolicy_titlecells/` at commit `70b8e8f7c81a7d8d673286d0da918eeae6424dc1` of our internal repository. 22 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
