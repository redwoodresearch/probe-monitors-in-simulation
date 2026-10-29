# 2026-10-12_q3_nano_realsim

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does the simulated control evaluations paper's gpt-5.4-nano transcript monitor catch simulated attack runs as often as real ones on the reply detector's 939 cells?

## What we did

\$0, the paper's stored scores: gate reproduced the paper's nano summary, then compared detection at 5% and 1% budgets (threshold from 1,000 real honest runs) for simulated, first real and second real runs, paired by cell with main-task resampling.

## What we found

README: at 5% second real minus simulated +3.1 points [+0.2, +6.1], second real minus first real -0.5 [-3.8, +2.8]; no refusals.

## In the post?

Not in the post.

## Status

Exploratory (post hoc, preliminary)

## Builds on

The simulated control evaluations paper's monitor scores; [2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/) (cells)

## Main scripts

- [`analyze_nano_realsim.py`](analyze_nano_realsim.py)
- [`analyze_nano_planted.py`](analyze_nano_planted.py)
- [`run_nano_onpolicy.py`](run_nano_onpolicy.py)
- [`analyze_nano_onpolicy.py`](analyze_nano_onpolicy.py)
- [`plant_both.py`](plant_both.py)

Result files the index names:

- `nano_realsim_POSTHOC_2026-10-12_125249.json` (not in this folder at the source commit)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_nano_realsim/` at commit `ff9b9ac96742ddfbbcbb79102156cee7ebad15b4` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
