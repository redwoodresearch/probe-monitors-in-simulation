# 2026-10-12_claimB_planted_oppoint

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

At the halves claim's own operating point (the frozen 1-in-100 alarm; halves: train on one half of the environments and test on the other, in both directions), does the simulation-trained detector flag copies with one simulator slip restored more often than their twins?

## What we did

Re-scored the 250 plant-2 pairs (one str_replace output reverted to the simulator's format) on Modal with each direction's selected detector, for Qwen3-32B and Qwen3.5-27B readers, and compared copy vs twin flag rates at the frozen alarm with 2,000 task resamples.

## What we found

Codex fact-check: Qwen3-32B detects the plant in both directions (pooled copies 36/250 vs twins 12/250, shift 9.6 points [4.4, 16.8]); Qwen3.5-27B not demonstrated (1/250 vs 3/250).

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/); [2026-10-11_claimB_halves_qwen35](../2026-10-11_claimB_halves_qwen35/)

## Main scripts

- [`run_POSTHOC_planted_scores_modal.py`](run_POSTHOC_planted_scores_modal.py)
- [`analyze_POSTHOC_planted_oppoint.py`](analyze_POSTHOC_planted_oppoint.py)

Result files the index names:

- `planted_oppoint_POSTHOC_2026-10-12_114022.json` (not in this folder at the source commit)
- `codex_factcheck_planted_oppoint_POSTHOC_2026-10-12_114022.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_planted_oppoint/` at commit `26851ed338c3b6297cd4a102ae4b914ef2a2c8d7` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
