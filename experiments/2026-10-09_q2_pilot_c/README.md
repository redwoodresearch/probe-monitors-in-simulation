# 2026-10-09_q2_pilot_c

2026-10-09 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Among attack calls that advanced the side task but that a command-text monitor misses (subtle calls), are there as many in simulated runs as real ones, and does the activation probe catch them?

## What we did

Five LLM judges via OpenRouter rated whether marked calls advanced the side task; subtle calls were those the text baseline missed; the Qwen3.5-27B impactful-command probe's catch was compared off-policy (GPT-5.6-Sol runs) and on-policy (Qwen3.5-27B, September simulator) with a pre-registered expansion.

## What we found

Off-policy, 5 real and 10 simulated calls rated subtle; the probe caught 0 and 1. On-policy the probe caught 4 of 34 real vs 3 of 30 simulated subtle calls, gap +0.018 [-0.133, +0.149] (preliminary).

## In the post?

Not in the post.

## Status

Current (exploratory)

## Builds on

[2026-10-06_q2_impactful_call_transfer](../2026-10-06_q2_impactful_call_transfer/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`build_pool.py`](build_pool.py)
- [`build_pool_onpolicy.py`](build_pool_onpolicy.py)
- [`run_judges.py`](run_judges.py)
- [`analyze_pilot_c.py`](analyze_pilot_c.py)
- [`analyze_expansion.py`](analyze_expansion.py)

Result files the index names:

- `pilot_c_2026-10-09_120612.json` (not in this folder at the source commit)
- `expansion_primary_2026-10-09_135134.txt` (not copied: lines over 300 characters)
- `manifest_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_q2_pilot_c/` at commit `05f03862c4882a39225a90864aa425546b1ffdd1` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
