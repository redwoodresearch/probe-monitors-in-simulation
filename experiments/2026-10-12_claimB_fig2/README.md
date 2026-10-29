# 2026-10-12_claimB_fig2

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Can the environment-halves realism figure (Figure 2 of the October 12 write-up outline; train on one half of the environments and test on the other, in both directions) be redrawn from raw data, beside other readers?

## What we did

Plot scripts read the halves result and the Llama-3.2-1B, TF-IDF and number/ID/whitespace-masked halves results by key from pinned commits, checked values against the outline image, and drew bars of the share of simulated and second-real runs flagged at the frozen 1% alarm.

## What we found

Four renders (fig2_halves 11:27; fig2_v2_compare 11:29, 12:08, 13:04); re-plots of existing results (probe pooled 51/939 simulated vs 15/939 second-real runs flagged at the 1% threshold).

## In the post?

Not in the post (figure for the 10-12 write-up outline; the post's Figure 2 is the later Qwen3.5 simulation detector).

## Status

Current (on a side branch only)

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/); [2026-10-12_claimB_1b_halves](../2026-10-12_claimB_1b_halves/); [2026-10-12_claimB_tfidf_halves](../2026-10-12_claimB_tfidf_halves/); [2026-10-12_claimB_halves_mask23](../2026-10-12_claimB_halves_mask23/)

## Main scripts

- [`plot_fig2_halves.py`](plot_fig2_halves.py)
- [`plot_fig2_v2_compare.py`](plot_fig2_v2_compare.py)

Result files the index names:

- `.png` (not in this folder at the source commit)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-12_claimB_fig2/` at commit `94efa78cb4d9321468327dca3a85b2cefb006be9` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
