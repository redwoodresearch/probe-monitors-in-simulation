# 2026-10-06_james_figures

2026-10-06 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Assemble a snapshot of first real-vs-simulated results as figures for the 2026-10-07 meeting with the project manager.

## What we did

Plotting script reading stored values from real-versus-simulated probe and text-classifier result files (no recomputation), producing four figures (probe vs TF-IDF text classifier by cleanup level, baselines, tool calls vs outputs, today vs September simulator) and a value table with sources; plus a small readout-newline count.

## What we found

Preliminary snapshot: on held-out environments probe and text classifier tie with no text cleanup (0.971 vs 0.971) and both drop with all seven cumulative cleanups of times and dates, numbers and IDs, whitespace, harness format, diff lines, empty diff lines and editor output format (0.541 vs 0.556).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-06_q1_segment_end](../2026-10-06_q1_segment_end/); [2026-10-06_q1_text_baselines](../2026-10-06_q1_text_baselines/); [2026-10-06_q2_impactful_call_transfer](../2026-10-06_q2_impactful_call_transfer/); [2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/) (config/sources.yaml)

## Main scripts

- [`plot_james_figures.py`](plot_james_figures.py)
- [`analyze_readout_newlines.py`](analyze_readout_newlines.py)

Result files the index names:

- `readout_newlines_2026-10-06_234901.json` (not copied: result left out to keep the release under 120 MB)
- `fig1_probe_vs_tfidf_by_level.png` (not copied: figure (not copied))
- `fig2_baselines_original_text.png` (not copied: figure (not copied))
- `fig3_calls_vs_outputs.png` (not copied: figure (not copied))
- `fig4_today_vs_september_simulator.png` (not copied: figure (not copied))

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-06_james_figures/` at commit `1ac63208009d972155c6ef5261055af65d28874a` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
