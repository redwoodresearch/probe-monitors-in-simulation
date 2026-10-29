# 2026-10-07_what_each_method_sees

2026-10-07 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

What do the TF-IDF text classifier and the Qwen3-32B probe each pick up when telling real from simulated runs with no text cleanup and with all seven cumulative cleanups (times and dates, numbers and IDs, whitespace, harness format, diff lines, empty diff lines, editor output format), and does the evidence add up over a run?

## What we did

Phase 1 diagnostic on saved out-of-fold scores for 939 paired attack cells: top-feature fits, disagreement examples and run-level accumulation, plus follow-ups on the honest real-versus-real floor, carry-over, attack batch matching and the residual after all seven cleanups.

## What we found

With no text cleanup both read the editor format (text classifier 0.792, probe 0.804); with all seven cleanups the leftover is a few rare simulator tells, and dropping five moves them to 0.514 and 0.524. The honest real-versus-real 0.553 is a batch split.

## In the post?

Not in the post.

## Status

Current (on a side branch)

## Builds on

[2026-10-06_q1_segment_end](../2026-10-06_q1_segment_end/); [2026-10-06_q1_text_baselines](../2026-10-06_q1_text_baselines/)

## Main scripts

- [`analyze_join.py`](analyze_join.py)
- [`run_top_features_modal.py`](run_top_features_modal.py)
- [`analyze_top_features.py`](analyze_top_features.py)
- [`analyze_adds_up.py`](analyze_adds_up.py)
- [`analyze_examples.py`](analyze_examples.py)
- [`plot_adds_up.py`](plot_adds_up.py)
- followups/\*.py

Result files the index names:

- [`join_2026-10-07_122336.json`](results/join_2026-10-07_122336.json)
- [`top_features_L0_2026-10-07_124121.json`](results/top_features_L0_2026-10-07_124121.json)
- [`top_features_L7_2026-10-07_124121.json`](results/top_features_L7_2026-10-07_124121.json)
- [`adds_up_2026-10-07_130053.json`](results/adds_up_2026-10-07_130053.json)
- [`honest_floor_2026-10-07_153859.md`](followups/results/honest_floor_2026-10-07_153859.md)

## Files here

15 code and configuration files and 7 result files, copied byte for byte from `experiments/2026-10-07_what_each_method_sees/` at commit `826291abbf6b900716547d46bb7e2c5ee32cfdb1` of our internal repository. 37 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
