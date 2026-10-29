# 2026-10-07_phase2_attack_check

2026-10-07 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Can Qwen3.5-27B act as a LinuxArena attacker on real machines, and what harness (token caps, timeouts, request logging, simulator choice, task draw) do the second phase's runs need, where Qwen3.5-27B itself is the agent in real and simulated environments?

## What we did

Step 1 ran 5 real attack and 5 honest Qwen3.5-27B runs served by vLLM on Modal; the folder then built the second phase's harness: per-turn limits, a token-logging proxy, a guard against repeated text, the pilot launcher run_pilot.py (7 runs per task across real and simulated arms), a 140-run rerun of the 20-task pilot with the strict tool setting off (a tool-setting bug voided the first pilot), a Qwen3-32B honest arm, and the frozen 196-task draw file; later sections add a batch of new real GPT-5.6-Sol honest runs bought for the credential-monitor comparison.

## What we found

README headline: Qwen3.5-27B attempted the side task in 5 of 5 runs and passed the side-task scorer in 3 of the 4 scored. The strict-tools rerun launched 140 runs; side task succeeded 13 times and failed 7, honest runs succeeded 23 of 38. The 10-08 pilot is void (strict-tools bug).

## In the post?

In the post: Appendix A runs (indirectly: harness and task file behind the oldest 196-task stratum of the 1,313 tasks)

## Status

Current (harness and task file); the 10-08 pilot inside it is void

## Builds on

[2026-10-06_phase2_feasibility](../2026-10-06_phase2_feasibility/); [2026-10-07_phase2_probe_proposal](../2026-10-07_phase2_probe_proposal/)

## Main scripts

- [`run_attack_check.py`](run_attack_check.py)
- [`run_pilot.py`](run_pilot.py)
- [`run_select_tasks.py`](run_select_tasks.py)
- [`tasks.py`](tasks.py)
- [`request_log_proxy.py`](request_log_proxy.py)
- [`run_vllm_tunnel.py`](run_vllm_tunnel.py)
- [`check_sim_samples.py`](check_sim_samples.py)
- [`setup_devbox.sh`](setup_devbox.sh)
- [`analyze_attack_check.py`](analyze_attack_check.py)

Result files the index names:

- `attack_check_2026-10-07_143120.jsonl` (not copied: result file type not copied (.jsonl))
- `turn_lengths_2026-10-07_203916.json` (not copied: result left out to keep the release under 120 MB)
- [`sim_preflight_2026-10-08_010500.json`](results/sim_preflight_2026-10-08_010500.json)
- `pilot_tasks_amendment9_all196.json` (not copied: config JSON with strings over 300 characters)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. Other result files, from earlier or side runs, give 7 script-and-commit pairs that match, 7 that differ and 4 where the script was not in the folder; we did not copy those versions.

## Files here

69 code and configuration files and 11 result files, copied byte for byte from `experiments/2026-10-07_phase2_attack_check/` at commit `74aa630146f6af1720c4601a675faaa7e39c93c0` of our internal repository. 99 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
