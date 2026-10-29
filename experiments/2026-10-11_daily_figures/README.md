# 2026-10-11_daily_figures

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Figures for the Monday daily meeting: the probe versus the text classifier on fixed-simulator runs, with per-bar intervals.

## What we did

Plot script reading committed result files with git show and value checks; a \$0 helper recomputes per-bar bootstrap intervals with the same task draws as the post hoc probe-versus-TF-IDF comparison.

## What we found

Figures only; probe_vs_tfidf_bars files reproduce the committed difference interval (gate in the script).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_claimB_q11_posthoc](../2026-10-11_claimB_q11_posthoc/); [2026-10-09_fixed_sim](../2026-10-09_fixed_sim/)

## Main scripts

- [`plot_daily_figures.py`](plot_daily_figures.py)
- [`analyze_probe_vs_tfidf_bars.py`](analyze_probe_vs_tfidf_bars.py)

Result files the index names:

- `probe_vs_tfidf_bars_2026-10-11_140001.json` (not copied: result left out to keep the release under 120 MB)
- `probe_vs_tfidf_bars_2026-10-11_141329.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_daily_figures/` at commit `2668a2dfb1ebcddd3eef816a59771117e4207dbe` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
