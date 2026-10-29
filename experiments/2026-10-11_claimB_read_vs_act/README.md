# 2026-10-11_claimB_read_vs_act

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

With the same reader (Qwen3.5-27B) and the same formatting fixes on both sides, how much does separation change between reading GPT-5.6-Sol's fixed runs (0.672) and Qwen3.5-27B's own runs (0.562)?

## What we did

Plan and power only: tabulated which fixes each side had (clock rule only on the reading side, create-newline rule only on the acting side) and priced re-reads; the project lead chose a README caveat instead, so nothing ran.

## What we found

PLAN.md: never ran; decision 'README caveat only, \$0' (relayed about 22:45 PT); the mismatch between the two sides is the clock rule (reading only) and the create-newline rule (acting only).

## In the post?

Not in the post.

## Status

Void (never ran; caveat added instead)

## Builds on

[2026-10-10_claimB_qwen35_reader](../2026-10-10_claimB_qwen35_reader/); [2026-10-10_claimB_onpolicy](../2026-10-10_claimB_onpolicy/)

## Main scripts

- [`power_read_vs_act.py`](power_read_vs_act.py)

Result files the index names:

- `results_manifest_v1.json` (not copied: JSON outside results/ and config/)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_read_vs_act/` at commit `6e2d08d173710cb68b2dab98e68045fb8256be1e` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
