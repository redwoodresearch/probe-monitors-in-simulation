# 2026-10-12_q2_jitter_530

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

At the 530-task off-policy monitor's 1% thresholds (Qwen3.5-27B reading GPT-5.6-Sol's runs, impactful-call probe), what false-alarm rate is achieved, how many scores tie at the threshold, and how were the intervals computed?

## What we did

\$0 local analysis of stored scores: achieved false alarms on the calibration sets, ties before and after jitter, refit-aggregate intervals and the real minus simulated difference on the same draws; a post hoc figure with fixed-probe intervals.

## What we found

Result JSON (first copy at 32836e29): achieved false alarms 0.90% (real) and 0.93% (simulated); 5 calibration negatives tie exactly at each threshold; real-trained real minus simulated catch +0.06 points [-1.07, +1.34].

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-08_impactful_refit_bootstrap](../2026-10-08_impactful_refit_bootstrap/); [2026-10-09_q2_tpr_at_1pct](../2026-10-09_q2_tpr_at_1pct/)

## Main scripts

- [`analyze_jitter_530.py`](analyze_jitter_530.py)
- [`plot_530_fixed_probe.py`](plot_530_fixed_probe.py)

Result files the index names:

- `jitter_530_2026-10-12_114651.json` (not in this folder at the source commit)
- `fig_530_fixed_probe_2026-10-13_v2.png` (not copied: result file type not copied (.png))

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q2_jitter_530/` at commit `1f0ee3ffae26bb51ef05c6a043ca064325ddde99` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
