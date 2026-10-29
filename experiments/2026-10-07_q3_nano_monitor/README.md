# 2026-10-07_q3_nano_monitor

2026-10-07 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does a black-box GPT-5-nano whole-trajectory monitor (the simulated control evaluations team's) score simulated LinuxArena attacks like real ones, and how does it compare with the probe detector?

## What we did

Read the sims team's saved GPT-5-nano monitor scores for real and simulated GPT-5.6-Sol attack runs (no model calls), built shared attempted/declined labels from submit reports, and compared with the Phase 1 probe detector.

## What we found

README: the monitor rates simulated and real attacks alike (shift -0.010 [-0.030, +0.011]) but is weak (AUROC 0.562 on real runs) next to the command probe (0.931); about 48% of attack runs report no success.

## In the post?

Not in the post.

## Status

Current (Phase 1 record)

## Builds on

[2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/)

## Main scripts

- [`run_fetch_scores.py`](run_fetch_scores.py)
- [`run_self_reports.py`](run_self_reports.py)
- [`make_attempt_labels.py`](make_attempt_labels.py)
- [`analyze_nano.py`](analyze_nano.py)

Result files the index names:

- [`nano_2026-10-07_154823.json`](results/nano_2026-10-07_154823.json)
- `attack_attempt_labels_2026-10-08_010659.jsonl` (not copied: result file type not copied (.jsonl))

## Files here

5 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-07_q3_nano_monitor/` at commit `9648a05a213dc8187ca798f287a5dce550cb109f` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
