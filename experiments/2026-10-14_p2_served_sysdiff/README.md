# 2026-10-14_p2_served_sysdiff

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

In the tokens vLLM actually served, how do the simulated and real run 1 system turns differ: only in the offered tool schemas, or also in prompt text?

## What we did

Counts-only Modal CPU alignment (difflib on token ids, never decoded) of the system turns of run 1 and simulated runs for 191 pilot tasks, splitting differing tokens into tool list, prompt text and special tokens.

## What we found

In all 191 tasks only the tool list differs (median 11 differing tokens out of about 890; prompt text differs in 0 tasks), from the simulator's relaxed tool schemas (RULE.md result).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_p2_dt_layout_diff](../2026-10-14_p2_dt_layout_diff/)

## Main scripts

- [`run_sysdiff.py`](run_sysdiff.py)

Result files the index names:

- `sysdiff_2026-10-14_182800.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_served_sysdiff/` at commit `dd12275923b0d5f873f3f0d034e4355b16584751` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
