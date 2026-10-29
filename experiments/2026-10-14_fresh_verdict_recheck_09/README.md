# 2026-10-14_fresh_verdict_recheck_09

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does the "fresh honest" verdict (how much more often the main probes false-alarm on honest runs of new tasks than on the original cohort) reproduce when two of its rows are recomputed with independent code?

## What we did

An independent agent recomputed on Modal CPU, from stored activations and frozen detector weights, the change in false alarms (new tasks minus cohort) with its own threshold, eligibility and bootstrap code for two rows: Qwen3-32B's simulation-trained probe (all formatting repairs plus the file-view newline fix, token-averaged readout, layer 48) and Seed-OSS-36B's real-trained probe (all formatting repairs, layer 48), both from the earlier phase on GPT-5.6-Sol transcripts.

## What we found

All four row-by-level values reproduced to within 3e-15 (634 runs, 317 tasks). Recomputed change in false alarms at 1%: Qwen3-32B +2.0 points (95% +0.7 to +3.4), Seed +1.0 (95% −0.03 to +2.2) (fresh_recheck_2026-10-14_081203.printout.txt).

## In the post?

Not in the post.

## Status

Current

## Builds on

The fresh-honest verdict (score_fresh_s1234567_2026-10-14_080417.json on a side branch)

## Main scripts

- [`recheck_fresh.py`](recheck_fresh.py)
- [`run_recheck_modal.py`](run_recheck_modal.py)

Result files the index names:

- [`fresh_recheck_2026-10-14_081203.json`](results/fresh_recheck_2026-10-14_081203.json)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_fresh_verdict_recheck_09/` at commit `e8cc1741e9a7881c9eb2a8b979d3466b166f7d99` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
