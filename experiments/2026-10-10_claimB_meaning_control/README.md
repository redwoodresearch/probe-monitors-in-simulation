# 2026-10-10_claimB_meaning_control

2026-10-10 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

In the fixed-simulator test (can probes tell simulated runs from real ones after the simulator's formatting mistakes are fixed?), does the frozen main realism probe respond to the meaning "this is a simulation", or only to simulator formatting mistakes?

## What we did

Planted three same-format lines (simulation, production system, neutral) in 150 real GPT-5.6-Sol attack runs and scored them with the frozen Qwen3-32B realism probe; a pre-registered amendment instead re-inserted one simulator formatting mistake (undoing the str_replace line-number fix) in a single tool output of the same runs as a matched check.

## What we found

Not shown to read meaning; the 'simulation' line moves the score slightly toward real (-0.0066), simulation minus production line -0.0029 [-0.0075, +0.0017]. One re-inserted formatting mistake moves run scores +0.0594 [0.0452, 0.0726] toward simulated (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-10_claimB_probe](../2026-10-10_claimB_probe/)

## Main scripts

- [`run_meaning_control_modal.py`](run_meaning_control_modal.py)
- [`run_meaning_plant_modal.py`](run_meaning_plant_modal.py)
- [`analyze_meaning_scale.py`](analyze_meaning_scale.py)
- [`undo_r2.py`](undo_r2.py)
- [`recheck_plant_matched_modal.py`](recheck_plant_matched_modal.py)

Result files the index names:

- `results_manifest_v1.json` (not copied: JSON outside results/ and config/)
- `results_manifest_v5.json` (not copied: JSON outside results/ and config/)

## Files here

9 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_claimB_meaning_control/` at commit `f3b4a08affce9c1f04cb8984fb122ebc1bcb4309` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
