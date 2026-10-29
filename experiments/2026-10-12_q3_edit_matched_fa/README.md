# 2026-10-12_q3_edit_matched_fa

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

With each world's alarm threshold set on its own honest runs, how much of the rendering edit's +13.7-point catch gain survives, and does edited simulated catch match real catch?

## What we did

\$0 reanalysis of the reply detector's stored scores with own-honest calibration in three cohorts (all 939 cells, shared tasks, task-disjoint non-pilot), with planted -3-point checks.

## What we found

README: about 0.61 of the frozen gain survives (+8.41 [+3.49, +11.45] vs +13.71 frozen); edited simulated vs second real is not readable (planted check not detected; -4.87 [-13.54, +3.82]).

## In the post?

Not in the post.

## Status

Exploratory (post hoc, preliminary); extended to Olmo in [2026-10-13_q3_reader_matched_fa](../2026-10-13_q3_reader_matched_fa/)

## Builds on

[2026-10-12_q3_edit_honest_fa](../2026-10-12_q3_edit_honest_fa/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`analyze_matched_fa.py`](analyze_matched_fa.py)
- [`analyze_matched_fa_overlap.py`](analyze_matched_fa_overlap.py)
- [`analyze_matched_fa_disjoint.py`](analyze_matched_fa_disjoint.py)
- [`count_outside_ci.py`](count_outside_ci.py)

Result files the index names:

- `manifest_results_v1.json` (not copied: strings over 300 characters)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_edit_matched_fa/` at commit `9d712105019c6b0eb028e8b6aa6b55979046d4ac` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
