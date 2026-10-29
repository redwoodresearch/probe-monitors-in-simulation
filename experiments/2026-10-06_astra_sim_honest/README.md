# 2026-10-06_astra_sim_honest

2026-10-06 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Can we generate simulated honest LinuxArena runs (GPT-5.6-Sol as the agent, GPT-6-Astra as the simulator, September simulator code) matched to the September simulated attack cohort, so the Phase 1 real-versus-simulated analyses can also run on honest transcripts? A side pilot asked how today's simulator code differs from September's.

## What we did

Rebuilt the September simulator code on a devbox and ran 437 simulated honest runs (one per main task, each forked from a real honest reference that is not in the comparison cohort), plus a 20-task pilot of today's simulator code (sims main e561f936) on honest and attack runs. Its runnable September-simulator snapshot (sims ba990399 plus the S3 archive plus a build patch, on the sims-astra devbox) is the simulator code the Phase 2 Qwen3.5-27B runs later used.

## What we found

README status: 437 of 437 runs valid, recorded spend \$3,293.06. Today's simulator renders editor diffs like real runs (1.0 vs September 0.087 vs real 1.0). Simulated honest runs copy a reference output verbatim for 15-17% of tool outputs vs 0-6% for attack runs.

## In the post?

Not in the post.

## Status

Current (Phase 1 data; feeds the Phase 1 honest-arm analyses; its simulator snapshot build was reused by [2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/))

## Builds on

None stated (matched to the September simulated attack cohort la_public10_sim_astra_attack_n1000)

## Main scripts

- [`run_sim_honest.py`](run_sim_honest.py)
- [`run_select_references.py`](run_select_references.py)
- [`run_build_references.py`](run_build_references.py)
- [`run_reproduce_contracts.py`](run_reproduce_contracts.py)
- [`run_today_pilot_setup.py`](run_today_pilot_setup.py)
- [`analyze_runs.py`](analyze_runs.py)
- [`analyze_editor_format.py`](analyze_editor_format.py)
- [`analyze_tool_outputs.py`](analyze_tool_outputs.py)

Result files the index names:

- `tool_outputs_20261006-185513.json` (not copied: strings over 300 characters)
- [`september_reads_20261006-144056.json`](results/september_reads_20261006-144056.json)
- [`today_pilot_tasks_20261006-175715.json`](results/today_pilot_tasks_20261006-175715.json)

## Files here

11 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-06_astra_sim_honest/` at commit `a2546a035a65fd16ece4c03241721030fe326dec` of our internal repository. 32 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
