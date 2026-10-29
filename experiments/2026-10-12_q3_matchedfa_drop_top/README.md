# 2026-10-12_q3_matchedfa_drop_top

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does the equal-false-alarm real minus simulated catch comparison change when the tasks that drive the simulated false alarms are dropped?

## What we did

\$0: copied the 196-task matched-false-alarm analysis with a task filter (gate: reproduces the original exactly) and dropped the top 1, 3 or 5 tasks by simulated false alarms.

## What we found

README: at 1% real minus simulated goes from +5.8 [-6.3, +10.6] (none dropped) to +2.6 (top 1) and -5.0 [-9.7, +1.5] (top 3); at 5% from +1.9 to -0.5 [-1.5, +0.4] (top 3).

## In the post?

Not in the post.

## Status

Exploratory (post hoc)

## Builds on

[2026-10-12_q3_fa_concentration](../2026-10-12_q3_fa_concentration/); the 196-task matched-false-alarm analysis (analyze_q22_matched_fa_POSTHOC.py; folder not stated)

## Main scripts

- [`analyze_q22_matched_fa_dropped_POSTHOC.py`](analyze_q22_matched_fa_dropped_POSTHOC.py)
- [`compute_drop_sets.py`](compute_drop_sets.py)
- [`check_gate.py`](check_gate.py)

Result files the index names:

- `drop_sets_POSTHOC_2026-10-12_104236.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_droptop_v1.json` (not copied: strings over 300 characters)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_matchedfa_drop_top/` at commit `83a05bf128e504db5d619d45b3fd47ead96c15b4` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
