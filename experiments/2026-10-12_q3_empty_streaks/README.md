# 2026-10-12_q3_empty_streaks

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do runs with long streaks of empty tool outputs (a simulator failure mode) carry a disproportionate share of the command monitor's false alarms?

## What we did

\$0, metadata only: longest streak of empty tool outputs per run in the 196-task acting runs (first real, simulated, second real, honest) and the reading set-up's runs, and the share of the frozen 1% false alarms inside runs with streaks of 20 or more and 50 or more.

## What we found

Result JSON and Codex recompute: 4 of 191 simulated acting runs have streaks of 20+ (max 128) vs 0 first-real and 1 second-real; they hold 122 of 189 simulated false alarms (64.6%); without them the simulated rate falls from 2.92% to 1.13%; no streaks in the reading set-up.

## In the post?

Not in the post.

## Status

Exploratory (post hoc; on a side branch only)

## Builds on

[2026-10-12_q3_fa_concentration](../2026-10-12_q3_fa_concentration/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`analyze_empty_streaks.py`](analyze_empty_streaks.py)

Result files the index names:

- `empty_streaks_POSTHOC_2026-10-12_131328.json` (not copied: result left out to keep the release under 120 MB)
- `codex_recompute.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_empty_streaks/` at commit `31020c5486df0135639ca9a1a68674bef28a400e` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
