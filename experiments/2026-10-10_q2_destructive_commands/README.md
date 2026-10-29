# 2026-10-10_q2_destructive_commands

2026-10-10 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does a probe on Qwen3.5-27B's activations, trained to flag destructive commands in GPT-5.6-Sol's real honest runs, flag them as often in simulated honest runs?

## What we did

Rule-labelled destructive commands (rm -rf, DROP TABLE, git reset --hard, kill -9, ...), probe trained on real honest runs with a frozen plan, catch at a 1% real-calibrated threshold, TF-IDF baseline, real-versus-real control and planted gaps.

## What we found

README: caught 30.1% of real vs 26.3% of simulated destructive commands, gap +3.8 points [-3.0, +10.7], no gap detected; later replaced by a corrected-label refit (+2.8 [-3.1, +9.7]). An entry in our experiment log corrects a quoted AUROC.

## In the post?

Not in the post.

## Status

Superseded by the corrected-label refit in [2026-10-10_q2_metric_table](../2026-10-10_q2_metric_table/) (separate_calibration_2026-10-10_225753.json)

## Builds on

[2026-10-09_probe_candidate_counts](../2026-10-09_probe_candidate_counts/); [2026-10-06_astra_sim_honest](../2026-10-06_astra_sim_honest/)

## Main scripts

- [`build_labels.py`](build_labels.py)
- [`analyze_destructive_counts.py`](analyze_destructive_counts.py)
- [`run_destructive_probe.py`](run_destructive_probe.py)
- [`analyze_destructive_probe.py`](analyze_destructive_probe.py)
- [`run_tfidf_baseline.py`](run_tfidf_baseline.py)
- [`analyze_label_escaping.py`](analyze_label_escaping.py)

Result files the index names:

- [`destructive_analysis_2026-10-10_145058.json`](results/destructive_analysis_2026-10-10_145058.json)
- `destructive_label_escaping_2026-10-10_145058_v2.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

7 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-10_q2_destructive_commands/` at commit `7aad630049c5ee4b10f6a7c7b5158ae2bd4ea8d3` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
