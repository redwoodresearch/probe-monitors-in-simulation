# 2026-10-08_q3_swap_llm_monitor

2026-10-08 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Do the two formatting fixes that move the tool-output probe also move the simulated control evaluations paper's LLM trajectory monitor (GPT-5.4-nano)?

## What we did

Ran the paper's GPT-5.4-nano whole-trajectory monitor on the 100 swap runs, as simulated vs with both fixes (600 calls, \$1.81), and checked whether its reasoning mentions the fixes; predictions first.

## What we found

No measured change (+3.0 points [-2.9, +8.9] detected at a 5% audit budget; simulated 23% vs real 25.0% / 22.2%), while the tool-output probe moved +20.5 points; 0 of 60 reasonings mention the gutter, line numbers or review note (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/); [2026-10-07_q2_rendering_swap](../2026-10-07_q2_rendering_swap/)

## Main scripts

- [`run_monitor_swap.py`](run_monitor_swap.py)
- [`analyze_monitor_swap.py`](analyze_monitor_swap.py)
- [`analyze_nano_reasoning.py`](analyze_nano_reasoning.py)

Result files the index names:

- `monitor_swap_analysis_2026-10-08_125143.json` (not copied: result left out to keep the release under 120 MB)
- `monitor_scores_manifest_2026-10-08.json` (not copied: result left out to keep the release under 120 MB)
- `nano_reasoning_manifest_2026-10-08_163503.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_q3_swap_llm_monitor/` at commit `9f8631f5eb696a6e6b84cb62abd94f60c3770b55` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
