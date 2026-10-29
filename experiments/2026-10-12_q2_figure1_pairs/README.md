# 2026-10-12_q2_figure1_pairs

2026-10-12 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Which real honest reply and simulated twin (one str_replace edit, one insert) should illustrate Figure 1 of the 10-12 write-up?

## What we did

\$0 local script that drew, by a stated seed, one same-state str_replace pair and one insert pair from real vs simulated honest runs (raw text kept out of git), plus an addendum counting which editor-format slips each candidate pair differs in.

## What we found

Selection and slip-count files with IDs and hashes only, over tier-2 pools of 130 str_replace and 9 insert pairs.

## In the post?

Not in the post (example for the 10-12 write-up outline).

## Status

Current

## Builds on

[2026-10-12_claimB_meaning_judge](../2026-10-12_claimB_meaning_judge/) (same-state convention); [2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (slip definitions)

## Main scripts

- [`select_figure1_pairs.py`](select_figure1_pairs.py)
- [`count_figure1_slips.py`](count_figure1_slips.py)

Result files the index names:

- [`figure1_selection_2026-10-12_150044.json`](results/figure1_selection_2026-10-12_150044.json)
- [`figure1_slip_counts_2026-10-12_175229.json`](results/figure1_slip_counts_2026-10-12_175229.json)

## Files here

2 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-12_q2_figure1_pairs/` at commit `cb23df891b7cc53015c997d737bf6a5bc8811c74` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
