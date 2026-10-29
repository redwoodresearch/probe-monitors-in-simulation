# 2026-10-09_q2_scorer_check

2026-10-09 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Did the same LinuxArena scorer grade GPT-5.6-Sol's and Qwen3.5-27B's honest runs (for the figure 1 competence comparison)?

## What we did

Read every honest run's Inspect log on both sides (39 Sol, 40 Qwen3.5-27B) and compared scorer settings, revisions, environment digests and scorer history; predictions first; \$0.

## What we found

Scorer settings identical and no task's scorer changed beyond a registration refactor since September, but content fingerprints differ on all 20 tasks, so whether the scorer code differs is unresolved (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-08_q2_figure_pack](../2026-10-08_q2_figure_pack/); [2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/)

## Main scripts

- [`analyze_scorer_versions.py`](analyze_scorer_versions.py)
- [`analyze_env_versions.py`](analyze_env_versions.py)
- [`analyze_scorer_history.py`](analyze_scorer_history.py)

Result files the index names:

- `scorer_versions_2026-10-09_103947.json` (not in this folder at the source commit)
- `manifest_v1.json` (not copied: strings over 300 characters)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_q2_scorer_check/` at commit `f363a07682839c15de48d13ab5204a25ae97f589` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
