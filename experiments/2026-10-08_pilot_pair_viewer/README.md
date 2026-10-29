# 2026-10-08_pilot_pair_viewer

2026-10-08 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Does a simulated Phase 2 run read like its real twin, step by step?

## What we did

A read-only side-by-side web viewer (on the sims-astra devbox, behind Tailscale) showing a simulated run beside its real twin for the same task, pairs drawn with a shown seed, with invalid runs flagged by a banner.

## What we found

Tool only, no scores; marks the broken 10-08 pilot (2026-10-08_124444_pilot) invalid.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (runs)

## Main scripts

- [`serve_devbox.sh`](serve_devbox.sh)
- [`sync_devbox.sh`](sync_devbox.sh)
- `src/probe_shift/viewer/pairs.py` (not in this folder at the source commit)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_pilot_pair_viewer/` at commit `7717b1faac18dca8cb0754a674f845770e258f7d` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
