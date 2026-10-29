# 2026-10-13_q3_sabotage_readers

2026-10-13 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does the sabotage training grid (real-trained, simulated-trained, mixed and reverse-mixed) give the same answers when Seed-OSS-36B, Olmo-3.1-32B, Llama-3.3-70B or Qwen3-32B reads Qwen3.5-27B's acting runs?

## What we did

Rendered the 518 runs per reader, located each command value, extracted last-token and mean readouts at 7 layers (Modal GPU), joined the project lead's frozen sabotage votes, and fit and scored the grid at each reader's primary layer, beside Qwen3.5-27B on the same calls; training-cell amendment compared simulated-trained with count-matched real-trained probes on 132 held-out tasks.

## What we found

training_cell_2026-10-13_215623.md: at 5%, simulated-trained minus real-trained catch is positive for most readable cells (Qwen3-32B layer 39 last token +8.7 [+2.0, +15.9]; Seed layer 40 mean +10.3; Olmo layer 32 paired -0.2), read as 'extends the recipe to sabotage'; Qwen3.5 self-read -15.4 but not readable; many 1% cells fail validity because the 32,768-token cut leaves 58 to 68% of command rows. Preliminary, labels unaudited.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sabotage_scaleup](../2026-10-13_q3_sabotage_scaleup/) (Amendment 3 grid); [2026-10-13_q3_qwen3_crossread](../2026-10-13_q3_qwen3_crossread/) (structural positions)

## Main scripts

- [`stage_readers.py`](stage_readers.py)
- [`run_extract_readers.py`](run_extract_readers.py)
- [`make_grid_spec.py`](make_grid_spec.py)
- [`run_grid_readers.py`](run_grid_readers.py)
- [`run_grid_q35_samecalls.py`](run_grid_q35_samecalls.py)
- [`run_analysis_modal.py`](run_analysis_modal.py)
- [`analyze_readers.py`](analyze_readers.py)
- [`report_readers.py`](report_readers.py)
- [`training_cell.py`](training_cell.py)

Result files the index names:

- [`report_readers_2026-10-13_185100.md`](results/report_readers_2026-10-13_185100.md)
- `training_cell_2026-10-13_215623.json` (not copied: result left out to keep the release under 120 MB)
- [`gate_qwen3_2026-10-13_164525.json`](results/gate_qwen3_2026-10-13_164525.json)

## Files here

21 code and configuration files and 5 result files, copied byte for byte from `experiments/2026-10-13_q3_sabotage_readers/` at commit `843b2c029b19234e01cf1aaead037ce97d96926a` of our internal repository. 36 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
