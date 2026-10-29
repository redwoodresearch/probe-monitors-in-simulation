# 2026-10-14_q3_new_mains_mode

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

The 360 new-main cells have no GPT-5.6-Sol sample, so the legacy launcher refuses them. How can real and simulated attacks be built for them from one record, without touching the legacy path?

## What we did

Prepared an additive construction mode (new_mains_v1) for the Phase-2 attack launcher: new entry points, a harness patch, a cells file built from git reads, provenance recording and tests; frozen before the scale-up's result.

## What we found

Mode prepared and frozen (per the experiment's written rules, nothing ran except make_new_mains_cells.py); used by [2026-10-14_q3_new_mains_launch](../2026-10-14_q3_new_mains_launch/).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_q2_scaleup_cohort](../2026-10-14_q2_scaleup_cohort/)

## Main scripts

- [`make_new_mains_cells.py`](make_new_mains_cells.py)
- [`record_provenance.py`](record_provenance.py)
- [`test_new_mains_mode.py`](test_new_mains_mode.py)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_q3_new_mains_mode/` at commit `ee9ba0263c6d1d79dcc9348eb41017a4dfe14501` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
