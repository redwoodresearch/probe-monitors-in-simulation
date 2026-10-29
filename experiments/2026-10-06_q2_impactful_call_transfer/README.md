# 2026-10-06_q2_impactful_call_transfer

2026-10-06 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do probes that flag impactful (not read-only) tool calls, trained on real LinuxArena runs, still work on simulated runs, and the reverse?

## What we did

Phase 1: a rule-based labeler for impactful calls, then Qwen3-32B probes (8 and 64 layers) at three readouts (end of the call, the template token just before the call, the previous content token) trained on real or simulated GPT-5.6-Sol runs in a 2x2 train/test design, with text baselines and follow-ups locating the shift (editor output rendering, insert review note, run history, behavior matching).

## What we found

README headline: read at the end of the call the probe transfers (recall at a frozen 1% threshold 0.893 real vs 0.894 simulated); read just before the call recall falls (0.218 vs 0.141), traced to how the September simulator renders text-editor outputs.

## In the post?

Not in the post.

## Status

Current (Phase 1 record)

## Builds on

[2026-10-06_segment_end_extraction](../2026-10-06_segment_end_extraction/); [2026-10-06_all_layers_extraction](../2026-10-06_all_layers_extraction/); [2026-10-06_astra_sim_honest](../2026-10-06_astra_sim_honest/) (simulated honest arm)

## Main scripts

- [`run_label_calls.py`](run_label_calls.py)
- [`run_q2_probes.py`](run_q2_probes.py)
- [`run_q2_alllayers.py`](run_q2_alllayers.py)
- [`run_q2_followups.py`](run_q2_followups.py)
- [`analyze_q2_final.py`](analyze_q2_final.py)
- [`analyze_text_baselines.py`](analyze_text_baselines.py)
- shift/\*.py

Result files the index names:

- `q2_probes_2026-10-06_215145.json` (not copied: result left out to keep the release under 120 MB)
- `q2_alllayers_2026-10-07_013628.json` (not copied: result file 1 MB or larger)
- `q2_final_table_2026-10-07_095958_056930.md` (not copied: lines over 300 characters)
- [`shift_stage1_2026-10-07_160508.json`](results/shift_stage1_2026-10-07_160508.json)
- [`shift_stage3_2026-10-07_160642.json`](results/shift_stage3_2026-10-07_160642.json)
- [`shift_str_replace_levels_2026-10-08_003204.json`](results/shift_str_replace_levels_2026-10-08_003204.json)

## Files here

44 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-06_q2_impactful_call_transfer/` at commit `bd004a0b010faff895abef666257e97e03d8aa6b` of our internal repository. 110 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
