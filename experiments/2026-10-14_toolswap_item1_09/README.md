# 2026-10-14_toolswap_item1_09

2026-10-14 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

How much of a positive-control probe's large simulated-vs-real gap comes from the 5–13 tool-schema tokens in the system turn rather than the simulated content?

## What we did

Rescored stored activations of simulated runs whose relaxed tool lines were replaced by the real ones (and a mirror on real runs) with frozen layer-63 models of three recipes, inside the segment-end pipeline, after the original tie-back gate failed for a staging difference.

## What we found

For the recipe trained on simulated attacks versus real honest runs at 1% the gap is 26.8 points (95% +20.5 to +33.3); swapping in the real tool lines lowers it by only 1.6 (95% −0.5 to +3.8), about 6% of the gap: "explains less than half" (toolswap_within_2026-10-14_133329.json).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-11_sim_toollist_swap](../2026-10-11_sim_toollist_swap/) (Row 4 stagings)

## Main scripts

- [`run_toolswap_item1.py`](run_toolswap_item1.py)
- [`run_toolswap_within.py`](run_toolswap_within.py)

Result files the index names:

- `toolswap_item1_2026-10-14_131443.json` (not copied: result left out to keep the release under 120 MB)
- `toolswap_within_2026-10-14_133329.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_toolswap_item1_09/` at commit `6648bc428230cb7c8f2cd3bfcab6f6fe71452c0b` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
