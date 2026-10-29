# 2026-10-14_p2_dt_layout_diff

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Why do the call-level monitors for destructive and transfer commands score the same calls differently in the served-token layout (the tokens vLLM actually served the policy) and in the reader layout used for the scale-up?

## What we did

A counts-only Modal CPU comparison of token prefixes for 44,582 calls present in both layouts: the first point of difference, differing tokens by category (system, reasoning, content), and whether the score change grows with the number of differences or with prefix length.

## What we found

No call had a byte-identical prefix (0 of 44,582); every run differs from the system turn on (no offered-tool block in the reader render, think blocks rendered differently); score changes did not grow with differences or length (R² 0.001–0.022) (RULE.md result section; independent recheck EXACT).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_p2_dt_scaleup](../2026-10-14_p2_dt_scaleup/) (failed bridge to the 196 pilot tasks)

## Main scripts

- [`run_layout_diff.py`](run_layout_diff.py)

Result files the index names:

- `layout_diff_2026-10-14_181042.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_dt_layout_diff/` at commit `555e8f9bb6f092f0f299b1bc91a2c6e682bbfc5f` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
