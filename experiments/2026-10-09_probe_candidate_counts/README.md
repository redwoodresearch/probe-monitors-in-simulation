# 2026-10-09_probe_candidate_counts

2026-10-09 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do three candidate probes from OpenAI's coding-agent monitoring list (unauthorized data transfer, destructive actions, deception) have enough training examples?

## What we did

Free keyword and command-rule counts over GPT-5.6-Sol's Phase 1 call labels and Qwen3.5-27B's strict rerun, with planted examples to check the rules.

## What we found

GPT-5.6-Sol real attack runs show external transfers in 33 of 977 vs 5 of 1,000 honest runs, but Qwen's rerun has transfers in only 1 real attack and 1 real honest run; 2 of 11 planted examples misclassified (preliminary).

## In the post?

Not in the post.

## Status

Exploratory (on a side branch)

## Builds on

[2026-10-06_q2_impactful_call_transfer](../2026-10-06_q2_impactful_call_transfer/) (labeler table); [2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/)

## Main scripts

- [`count_candidates.py`](count_candidates.py)

Result files the index names:

- [`probe_candidates_2026-10-09_161732.json`](results/probe_candidates_2026-10-09_161732.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-09_probe_candidate_counts/` at commit `e95c8db146f4b4b3ec3db36aa540aeab71cbf388` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
