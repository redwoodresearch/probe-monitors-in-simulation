# 2026-10-12_monitor_forest

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Plotted together, how does every monitor's real minus simulated catch gap compare with its two-real-runs gap, grouped by whether it watches the agent or the simulated computer?

## What we did

\$0 re-plot of existing results: each value read by key from pinned files listed in rows.jsonl, signs flipped to one convention, drawn as an appendix forest chart and a main-text version for the 10-12 write-up.

## What we found

Most command-reading monitors differ about as much as two real runs; reply-reading detectors differ more (+10.36 as-is, +13.29 at matched false alarms, +8.44 with another reader, -0.88 after the formatting fix); 75 plotted values rechecked.

## In the post?

Not in the post (figure for the 10-12 write-up outline).

## Status

Current (on two side branches)

## Builds on

[2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/); [2026-10-12_q2_olmo_reader](../2026-10-12_q2_olmo_reader/) (and other sources in rows.jsonl)

## Main scripts

- [`plot_monitor_forest.py`](plot_monitor_forest.py)
- [`plot_monitor_forest_main.py`](plot_monitor_forest_main.py)

Result files the index names:

- `forest_2026-10-12_121349.png` (not copied: result file type not copied (.png))
- `forest_main_2026-10-12_123039.png` (not copied: result file type not copied (.png))
- `caption_main_2026-10-12_123039.md` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `rows.jsonl` (not copied: per-run records (.jsonl))

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_monitor_forest/` at commit `bd0d43dfa0a40a7deaacb339417f8b3eb88585e0` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
