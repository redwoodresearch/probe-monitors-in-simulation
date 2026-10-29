# 2026-10-06_transcript_viewer

2026-10-06 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

What does the reader model see in real and simulated LinuxArena runs of the same task?

## What we did

A local web viewer (with a cache builder and a devbox-hosted copy) showing, per task, the real attack run, simulated attack run, second real attack run and real honest run side by side as the Qwen3-32B reader receives them, with the 32k cut marked and slots for probe scores.

## What we found

Tool built and checked against all 4,001 samples; about half of end-of-tool-call readout tokens are the final 0 of a fixed timeout 120 setting (50% in each attack arm, 39% in honest runs).

## In the post?

Not in the post.

## Status

Current

## Builds on

Nothing earlier in this index.

## Main scripts

- [`run_build_cache.py`](run_build_cache.py)
- [`run_viewer.sh`](run_viewer.sh)
- [`serve_devbox.sh`](serve_devbox.sh)
- [`sync_devbox.sh`](sync_devbox.sh)

Result files the index names:

- [`build_2026-10-06_172031.json`](results/build_2026-10-06_172031.json)

## Files here

5 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-06_transcript_viewer/` at commit `d38f42eb113891c9653000fc45b2e6c476333b3d` of our internal repository. 20 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
