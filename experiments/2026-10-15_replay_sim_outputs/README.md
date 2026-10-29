# 2026-10-15_replay_sim_outputs

2026-10-15 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

How much of the real-trained probes' simulated-vs-real gap comes back if we keep a real run's actions and only rewrite its tool outputs with the simulator?

## What we did

Pre-registered replay design (phase 1: 90 pilot estimation tasks; placebo on honest runs), join gate, and analysis of replayed vs real vs simulated catch for six probes at both readouts; reproduction gates passed.

## What we found

Phase 1, 90 tasks, tool outputs 5%: on-policy (training-fraction 100%) replay minus real -4.44 vs simulated minus real -10.00; off-policy +0.00 vs -5.56; gap unresolved; phase 2 not run (analysis printout).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-15_q2_replay](../2026-10-15_q2_replay/)

## Main scripts

- [`analyze_replay.py`](analyze_replay.py)
- [`check_join_gate.py`](check_join_gate.py)

Result files the index names:

- `replay_analysis_2026-10-15_215248.json` (not in this folder at the source commit)
- `join_gate_2026-10-15_214818.json` (not in this folder at the source commit)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_replay_sim_outputs/` at commit `0df0464d704e07c3f7248889808fa0f39de84703` of our internal repository. 15 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 2 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
