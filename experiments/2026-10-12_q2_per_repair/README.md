# 2026-10-12_q2_per_repair

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Which editor repair (the insert review note, or the str_replace numbered gutter) closes the reply detector's gap when applied alone?

## What we did

First part (\$0): stored single-repair arms on the rendering swap's 100 cells for Qwen3-32B. Second part: staged each repair alone on all 939 cells (817 str_replace, 427 insert runs), re-extracted Qwen3-32B and Olmo-3.1-32B activations on Modal and scored with frozen detectors (\$43.84).

## What we found

First-part result: at 5% the insert repair alone closes 36% of the as-is gap, the str_replace repair alone 71%, both together 117% (100 cells). The second part's reading is on S3 only and was not read here.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/); [2026-10-12_q2_olmo_reader](../2026-10-12_q2_olmo_reader/)

## Main scripts

- [`analyze_per_repair_100.py`](analyze_per_repair_100.py)
- [`run_stage_per_repair.py`](run_stage_per_repair.py)
- [`run_extract_qwen3_per_repair.py`](run_extract_qwen3_per_repair.py)
- [`run_extract_olmo_per_repair.py`](run_extract_olmo_per_repair.py)
- [`run_score_per_repair_qwen3.py`](run_score_per_repair_qwen3.py)
- [`run_score_per_repair_olmo.py`](run_score_per_repair_olmo.py)
- [`analyze_per_repair_939.py`](analyze_per_repair_939.py)

Result files the index names:

- `per_repair_100_2026-10-12_133342.json` (not copied: result left out to keep the release under 120 MB)
- [`per_repair_olmo_scores_sr_2026-10-12_145740.json`](results/per_repair_olmo_scores_sr_2026-10-12_145740.json)
- [`per_repair_olmo_scores_insert_2026-10-12_145721.json`](results/per_repair_olmo_scores_insert_2026-10-12_145721.json)
- [`stage_per_repair_2026-10-12_133921.json`](results/stage_per_repair_2026-10-12_133921.json)

## Files here

9 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-12_q2_per_repair/` at commit `017b692f75a2a6ee042e3d924afddea1afa4eec3` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
