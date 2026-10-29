# 2026-10-12_q2_olmo_reader

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does the reply-reading break replicate with Olmo-3.1-32B as a third reader, and do the two formatting fixes explain it?

## What we did

Tokenizer-only render gate on CPU, then extracted Olmo-3.1-32B activations for 4,630 transcripts on Modal on the same outputs Qwen3-32B kept (its readout rule differs: it skips newline-bearing tokens), refit the reply detector's recipe, scored simulated vs second real runs, and rescored the formatting-fixed runs with frozen probes; label-shuffle null check.

## What we found

README: "replicates": simulated minus second real at 5% -8.65 points [-10.94, -4.76]; the fixes "explain it" (fixed gap +2.08 [-0.84, +4.87], about 76% closed). Detector AUROC 0.944 vs selection-aware null max 0.564.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`run_olmo_render.py`](run_olmo_render.py)
- [`run_extract_olmo.py`](run_extract_olmo.py)
- [`run_score_fix_olmo.py`](run_score_fix_olmo.py)
- [`analyze_reply_break_olmo.py`](analyze_reply_break_olmo.py)
- [`analyze_edit_effect_olmo.py`](analyze_edit_effect_olmo.py)
- [`check_label_shuffle_null_olmo.py`](check_label_shuffle_null_olmo.py)

Result files the index names:

- `reply_break_olmo_2026-10-12_113245.json` (not in this folder at the source commit)
- `edit_effect_olmo_2026-10-12_115023.json` (not in this folder at the source commit)
- `render_reading_2026-10-12_044539.json` (not in this folder at the source commit)
- `label_shuffle_null_check_olmo_2026-10-13_045646.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

12 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q2_olmo_reader/` at commit `974c82639e5e1276ca526fb70859a3c5de508c5c` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
