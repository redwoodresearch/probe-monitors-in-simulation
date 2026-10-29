# 2026-10-06_phase2_feasibility

2026-10-06 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Can Qwen3-32B act as a LinuxArena agent well enough for on-policy probing (Phase 2), how does Qwen3.5-27B compare, and what would a batch cost?

## What we did

Self-hosted Qwen3-32B and Qwen3.5-27B on Modal H200 vLLM servers as the real LinuxArena honest agent on 5 tasks GPT-5.6-Sol had solved (General Sandbox devbox, sce arena policy), with smoke tests and a batch cost estimate.

## What we found

README headline: Qwen3-32B solved 0 of 5 tasks (1 of 5 without YaRN), Qwen3.5-27B 5 of 5 (not a success rate at n = 5); Qwen3.5-27B recommended; a Phase 2 batch estimated at about 9-21 thousand dollars, mostly the simulator.

## In the post?

Not in the post (but it chose Qwen3.5-27B, the post agent).

## Status

Current (its pilot runs are not in the post; the Phase 2 harness continued in [2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/))

## Builds on

Nothing earlier in this index.

## Main scripts

- [`run_smoke.py`](run_smoke.py)
- [`run_vllm_server.py`](run_vllm_server.py)
- [`run_pilot.py`](run_pilot.py)
- [`run_setup_key.py`](run_setup_key.py)
- [`analyze_pilot.py`](analyze_pilot.py)

Result files the index names:

- `pilot_2026-10-06_la_real_honest_phase2-qwen35_27b_2026-10-06_161341.jsonl` (not copied: result file type not copied (.jsonl))
- `pilot_2026-10-06_la_real_honest_phase2-qwen3_32b_2026-10-06_160649.jsonl` (not copied: result file type not copied (.jsonl))
- `pilot_2026-10-06_la_real_honest_phase2-qwen3_32b_native_2026-10-06_162947.jsonl` (not copied: result file type not copied (.jsonl))
- `smoke_Qwen3.5-27B_2026-10-06_151543.json` (not copied: strings over 300 characters)

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-06_phase2_feasibility/` at commit `d3ea1038f6aa065d1d8e2265bdd28ecd7599e98e` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
