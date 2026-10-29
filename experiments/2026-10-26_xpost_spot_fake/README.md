# 2026-10-26_xpost_spot_fake

2026-10-26 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Draw one real and one simulated honest-run tool output for a "spot the fake" social-media post, by a rule fixed in advance.

## What we did

Seeded draw of a real/simulated pair of first tool outputs from honest runs under frozen rule v1 then v2, plus a copy check against the simulator's reference run.

## What we found

Version 1 of the rule stopped on 19 duplicate .eval copies and was voided for a filter bug; version 2 drew a pair (t384_tiktok); the copy check found the simulated output identical to the reference run's output for the same command (spend rows; NOTE v2).

## In the post?

Not in the post.

## Status

Current (outreach material)

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) full-length inputs registry

## Main scripts

- [`run_xpost_spot_fake_v2.py`](run_xpost_spot_fake_v2.py)
- [`run_xpost_copycheck_v2b.py`](run_xpost_copycheck_v2b.py)
- [`run_xpost_spot_fake.py`](run_xpost_spot_fake.py)

Result files the index names:

- `xpost_spotfake_draw_v2_2026-10-26_163406.json` (not copied: result file 1 MB or larger)
- `xpost_copycheck_v2b_2026-10-26_210844.json` (not copied: result left out to keep the release under 120 MB)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-26_xpost_spot_fake/` at commit `547c1fd543a02d0cfd85cfea9e15214b9bb78265` of our internal repository. 20 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
