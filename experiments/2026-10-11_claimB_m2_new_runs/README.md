# 2026-10-11_claimB_m2_new_runs

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

What would new runs cost, and how many are needed, to settle two fixed-simulator-test items left as "too few to tell" (whether a richer probe beats the main realism probe at a 1% alarm, and the precision of hypothesis 1's output-swap test)?

## What we did

\$0 sizing: per-run API costs from 10 seeded logs per arm at the simulated-control-evals repository's listed rates, power for a fixed-threshold design, unused tasks in 10 environments of a separate 18-environment LinuxArena cohort, and the cost of a simulated honest run; nothing launched.

## What we found

PLAN.md and commits: real attack run \$0.523 each, simulated honest run \$7.86 each; the fixed-threshold design reaches 80% only if test-like at 110-440 cells, so the recommendation was not to buy; the yield pilot (about \$325 with buffer) was dropped when the project lead dropped the swap test (commit 9405b7475).

## In the post?

Not in the post.

## Status

Current (pricing only; nothing bought)

## Builds on

[2026-10-11_claimB_h2_probe_best_found](../2026-10-11_claimB_h2_probe_best_found/); [2026-10-11_claimB_h1_text_vs_layout](../2026-10-11_claimB_h1_text_vs_layout/)

## Main scripts

- [`analyze_run_costs.py`](analyze_run_costs.py)
- [`analyze_sim_honest_cost.py`](analyze_sim_honest_cost.py)
- [`analyze_power_new_runs.py`](analyze_power_new_runs.py)
- [`analyze_power_new_runs_v2.py`](analyze_power_new_runs_v2.py)
- [`analyze_power_fixed_threshold.py`](analyze_power_fixed_threshold.py)
- [`analyze_h1_triples_needed.py`](analyze_h1_triples_needed.py)

Result files the index names:

- `run_costs_2026-10-11_150334.json` (not copied: result left out to keep the release under 120 MB)
- `sim_honest_cost_2026-10-11_152605.json` (not copied: result left out to keep the release under 120 MB)
- `power_new_runs_2026-10-11_144510.json` (not copied: result left out to keep the release under 120 MB)
- `power_new_runs_v2_2026-10-11_145329.json` (not copied: result left out to keep the release under 120 MB)
- `power_fixed_threshold_2026-10-11_151207.json` (not copied: result left out to keep the release under 120 MB)
- `h1_triples_needed_2026-10-11_145556.json` (not copied: result left out to keep the release under 120 MB)

## Files here

6 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_m2_new_runs/` at commit `76fb0733c945044885564b063fa19d2cca3a1fbc` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
