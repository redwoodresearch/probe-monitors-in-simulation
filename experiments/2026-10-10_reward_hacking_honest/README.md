# 2026-10-10_reward_hacking_honest

2026-10-10 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Are there enough reward-hacking candidates (test edits, skipped or deleted tests, forced success) in honest runs, real and simulated, to justify a paid reward-hacking probe?

## What we did

Free rule-based screen over Qwen3.5-27B and GPT-5.6-Sol honest runs, real and simulated, with seeded audits that revised the rules.

## What we found

README: too few; strong candidates were rare (Qwen real 0 and 3, Sol real 15, Sol simulated 7), neither group clears the 200-command / 20-task bar, so no paid probe is recommended.

## In the post?

Not in the post.

## Status

Current (screen; no probe built)

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/); [2026-10-06_astra_sim_honest](../2026-10-06_astra_sim_honest/)

## Main scripts

- [`analyze_counts.py`](analyze_counts.py)
- [`analyze_candidates.py`](analyze_candidates.py)

Result files the index names:

- `counts_v4.json` (not copied: strings over 300 characters)
- `audit_v4.json` (not copied: result left out to keep the release under 120 MB)
- `archive_manifest.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_reward_hacking_honest/` at commit `3b5a777fea1a06f652481dce546f4cce744443a4` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
