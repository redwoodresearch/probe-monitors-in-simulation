# 2026-10-11_text_classifier_catch

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does a classifier that reads only each call's command text show the same real-versus-simulated catch as the 196-task impactful-command probe, at the probe's alarm level?

## What we did

Built text features of each call, trained a command-text classifier with the probe's folds, frozen 1% threshold and 500 task-resampled refit draws, with real run 2 as yardstick and planted checks; Modal.

## What we found

README: text classifier real minus simulated catch -1.21 points [-4.05, +1.84] vs the probe's +0.46 [-2.32, +2.50]; yardstick -1.17; AUROC 0.9917 real vs 0.9892 simulated.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`build_text_calls.py`](build_text_calls.py)
- [`run_text_catch.py`](run_text_catch.py)
- [`analyze_paired.py`](analyze_paired.py)

Result files the index names:

- [`scaleup196_verdicts_2026-10-11_201242.json`](results/scaleup196_verdicts_2026-10-11_201242.json)
- [`paired_probe_minus_text_2026-10-11_201242.json`](results/paired_probe_minus_text_2026-10-11_201242.json)
- `scaleup_plants_2026-10-11_201235.json` (not copied: result left out to keep the release under 120 MB)
- [`text_calls_summary_2026-10-11_192900.json`](results/text_calls_summary_2026-10-11_192900.json)

## Files here

3 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_text_classifier_catch/` at commit `253b245045ff61ca73a4db1a5da132cc01d0900c` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
