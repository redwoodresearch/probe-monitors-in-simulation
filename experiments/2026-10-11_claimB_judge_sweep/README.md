# 2026-10-11_claimB_judge_sweep

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does a judge prompt picked from a sweep on practice environments let GPT-5.6-Sol separate fixed-simulator runs from real ones as well as the probe?

## What we did

Swept judge prompts on practice environments through the batch API (plus a Gemini pilot), picked a prompt of three binary real-or-simulated questions, then scored 200 fixed-simulator and 200 real test runs; post hoc strict-alarm and side-by-side comparisons.

## What we found

README: picked prompt AUROC 0.531 vs probe 0.821 (difference -0.290 [-0.336, -0.238]), so the judge is at least 0.10 below the probe; the pick's shuffle-null p = 0.34; post hoc at 1% the judge catches 12 of 200 vs the probe's 4 (difference not established); side by side on practice pairs 0.71.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-11_claimB_judge](../2026-10-11_claimB_judge/)

## Main scripts

- [`run_sol_b.py`](run_sol_b.py)
- [`run_sol_b_big.py`](run_sol_b_big.py)
- [`analyze_sol_b.py`](analyze_sol_b.py)
- [`run_gemini_pilot.py`](run_gemini_pilot.py)
- [`analyze_sweep_sizing.py`](analyze_sweep_sizing.py)
- [`analyze_b_sizing.py`](analyze_b_sizing.py)

Result files the index names:

- `sol_b_test_2026-10-12_021302.json` (not copied: result left out to keep the release under 120 MB)
- `sol_b_pick_2026-10-11_231658.json` (not copied: result left out to keep the release under 120 MB)
- `alarm_ci_POSTHOC_2026-10-12_022750.json` (not copied: result left out to keep the release under 120 MB)
- `probe_paired_POSTHOC_2026-10-12_022804.json` (not copied: result left out to keep the release under 120 MB)
- `s3_ci_POSTHOC_2026-10-12_022123.json` (not copied: result left out to keep the release under 120 MB)
- `ties_lean_POSTHOC_2026-10-12_021313.json` (not copied: result left out to keep the release under 120 MB)

## Files here

11 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_claimB_judge_sweep/` at commit `b785818ea9628d87299fafec1f72a131f98a9fb7` of our internal repository. 17 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
