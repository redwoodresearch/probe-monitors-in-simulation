# 2026-10-12_q3_llama70b_price

2026-10-12 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Can Llama-3.3-70B act as a LinuxArena agent well enough to serve as another acting model (20-task honest check)?

## What we did

Served Llama-3.3-70B with vLLM on Modal H200s, smoke-tested (failed on tool-list placement; fixed; one retry failed loading weights; re-smoke passed), then ran 40 honest runs (20 tasks, 2 each) through the LinuxArena harness with a tool-output delivery check, under a \$160 stop.

## What we found

README: 1 of 40 main-task successes; every run ended in an early submit (median 2 steps); tool outputs were delivered byte-exact; predicted 3 plus or minus 3 successes and 30 plus or minus 15 steps.

## In the post?

Not in the post.

## Status

Exploratory (dead end: the model did not do the tasks)

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (harness and server configs)

## Main scripts

- [`analyze_smoke.py`](analyze_smoke.py)
- [`check_tool_outputs_llama.py`](check_tool_outputs_llama.py)
- [`combine_40.py`](combine_40.py)
- [`devbox/llama_full.sh`](devbox/llama_full.sh)
- [`watch_job.sh`](watch_job.sh)

Result files the index names:

- `combined_40_2026-10-12_162634.json` (not in this folder at the source commit)
- [`smoke_2026-10-12_150657.json`](results/smoke_2026-10-12_150657.json)
- [`tool_outputs_llama_2026-10-12_162610.json`](results/tool_outputs_llama_2026-10-12_162610.json)

## Files here

10 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-12_q3_llama70b_price/` at commit `89b9d26c1a44349b1d902ad8a3578d0904203643` of our internal repository. 15 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
