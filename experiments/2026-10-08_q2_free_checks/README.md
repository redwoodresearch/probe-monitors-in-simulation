# 2026-10-08_q2_free_checks

2026-10-08 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Four free checks on saved impactful-command scores: does the off-policy gap hold on the pilot's 20 tasks, how does a text classifier compare, does command length matter, and per environment and attempted-only?

## What we did

CPU analyses of saved per-call tables (no new fits), predictions committed first; batch 2 added attempted-only and per-environment off-policy gaps for Qwen3.5-27B.

## What we found

On the pilot tasks catch -1.9 points; a text classifier catches 96.9% vs the probe's 92.2%; catch 85-93% across command lengths. Attempted-only catch gap -0.39 points [-3.9, +3.2], no environment's interval excludes zero (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-08_impactful_call_2x2](../2026-10-08_impactful_call_2x2/)

## Main scripts

- [`analyze_free_checks.py`](analyze_free_checks.py)

Result files the index names:

- `manifest_v3.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_batch2_v2.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_q2_free_checks/` at commit `648c63e9f7da20902569076b6e1c953b91e8a495` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
