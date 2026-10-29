# 2026-10-13_q3_sabotage_scaleup

2026-10-13 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

With many more labelled tasks, does a sabotage alarm (a probe that flags the sabotage command itself, on Qwen3.5-27B's own activations) catch simulated sabotage at the same rate as real sabotage?

## What we did

Sized and priced the labelling, chose 116 new held-out tasks by coverage (the project lead labelled them with the five-judge panel, about \$258), scored the frozen stage-1 gate (layer 30, command end) on 132 held-out tasks in simulated, first and second real runs; then a training grid (real-trained, simulated-trained, mixed) and capacity analyses. Modal CPU.

## What we found

Codex recompute (codex/recompute_scaled_2026-10-13.md, all 627 fields match): at 1% on 132 held-out tasks, catch is 20.9% (first real), 18.8% (second real), 16.7% (simulated); simulated minus first real -4.16 [-11.90, +3.40] 'inconclusive'; second minus first real -2.10 'within 10 points'; simulated minus pooled real -3.11 [-9.98, +3.76] 'within 10 points'. Labels unaudited.

## In the post?

Not in the post.

## Status

Exploratory (labels not audited)

## Builds on

[2026-10-13_q3_sabotage_precision](../2026-10-13_q3_sabotage_precision/) (sizing); [2026-10-13_q3_sim_sabotage_ids](../2026-10-13_q3_sim_sabotage_ids/) (labels); [2026-10-12_q3_sabotage_adjudication](../2026-10-12_q3_sabotage_adjudication/) (stage-1 gate)

## Main scripts

- [`make_scaled_rows.py`](make_scaled_rows.py)
- [`launch_scaled.py`](launch_scaled.py)
- [`analyze_scaled.py`](analyze_scaled.py)
- [`launch_grid.py`](launch_grid.py)
- [`analyze_grid.py`](analyze_grid.py)
- [`analyze_xgrid.py`](analyze_xgrid.py)
- [`analyze_capacity_v2.py`](analyze_capacity_v2.py)
- [`check_scaled_stored.py`](check_scaled_stored.py)
- [`check_state_changing_only.py`](check_state_changing_only.py)
- [`build_packets_runs.py`](build_packets_runs.py)
- [`export_panel_votes_fresh.py`](export_panel_votes_fresh.py)
- [`price.py`](price.py)

Result files the index names:

- `coverage_summary_2026-10-13_125944.json` (not copied: result left out to keep the release under 120 MB)
- [`price_2026-10-13_130832.json`](results/price_2026-10-13_130832.json)
- `k116_label_counts_2026-10-13_152005.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `scaled_stored_check_2026-10-13_235009.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_scaled_s3_v3.json` (not copied: strings over 300 characters)
- `manifest_grid_s3_v2.json` (not copied: strings over 300 characters)

## Files here

29 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_q3_sabotage_scaleup/` at commit `c680f1fb2aba499d2b569b60e47d102a4097bc56` of our internal repository. 34 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
