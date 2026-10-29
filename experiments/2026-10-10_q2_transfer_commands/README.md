# 2026-10-10_q2_transfer_commands

2026-10-10 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Are there enough data-transfer commands in GPT-5.6-Sol's honest runs for a third action monitor, and does a Qwen3.5-27B probe trained on real runs flag them as often in simulated runs?

## What we did

A pattern count with a size rule committed first, then a probe on Qwen3.5-27B activations trained on real honest runs, scored on simulated honest runs at a real-calibrated 1% threshold, with a TF-IDF baseline.

## What we found

The size rule passes (823 transfer calls in 166 real tasks, 321 in 125 simulated). README: suggestive, not confirmed; real caught 0.710 vs simulated 0.642, gap +6.8 points [-1.0, +12.5]; the text classifier showed no gap.

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-09_probe_candidate_counts](../2026-10-09_probe_candidate_counts/); [2026-10-06_astra_sim_honest](../2026-10-06_astra_sim_honest/)

## Main scripts

- [`build_labels.py`](build_labels.py)
- [`analyze_transfer_counts.py`](analyze_transfer_counts.py)
- [`run_transfer_probe.py`](run_transfer_probe.py)
- [`analyze_transfer_probe.py`](analyze_transfer_probe.py)
- [`run_tfidf_baseline.py`](run_tfidf_baseline.py)
- [`check_rule_review.py`](check_rule_review.py)

Result files the index names:

- [`transfer_analysis_2026-10-10_213933.json`](results/transfer_analysis_2026-10-10_213933.json)
- [`transfer_probe_2026-10-10_213933.json`](results/transfer_probe_2026-10-10_213933.json)
- `rule_review_check_v2_2026-10-10.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

6 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-10_q2_transfer_commands/` at commit `101de42d349327bf0a8163091fe7c88aed31605b` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
