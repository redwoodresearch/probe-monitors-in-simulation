# 2026-10-11_la_format_sim_probe

2026-10-11 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Does a 'simulated vs production' probe trained on LinuxArena-format cue pairs transfer to LinuxArena runs better than the generic concept probe?

## What we did

Built 930 LinuxArena-format pairs with inserted simulation or production cues, extracted Qwen3.5-27B activations on one H200, fit content and turn-start probes, and compared held-out ranking with the concept probe; part 2 (real vs simulated) gated on a +0.05 improvement.

## What we found

README: closed; ranking score 0.531 vs the concept probe's 0.511 (+0.019, below the +0.05 bar), so part 2 was not run; \$5.08 billed.

## In the post?

Not in the post.

## Status

Exploratory (closed)

## Builds on

[2026-10-10_eval_awareness_probe](../2026-10-10_eval_awareness_probe/) (fit code and concept probe)

## Main scripts

- [`run_build_la_pairs.py`](run_build_la_pairs.py)
- [`run_extract_la.py`](run_extract_la.py)
- [`spawn_fit_la.py`](spawn_fit_la.py)
- [`analyze_la_compare.py`](analyze_la_compare.py)
- [`analyze_la_positions.py`](analyze_la_positions.py)
- [`plot_la_compare.py`](plot_la_compare.py)

Result files the index names:

- `compare_2026-10-11_124437.json` (not copied: result left out to keep the release under 120 MB)
- `compare_figure_2026-10-11_124719.png` (not copied: result file type not copied (.png))
- [`compare_scores_2026-10-11_124437.json`](results/compare_scores_2026-10-11_124437.json)

## Files here

6 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_la_format_sim_probe/` at commit `d2b569b5450d78a3804468548f469fbffb9950ae` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
