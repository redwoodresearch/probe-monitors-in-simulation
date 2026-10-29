# 2026-10-14_q2_b200_longrepro

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Do four activation dumps that ran partly on B200 GPUs, whose numerics break on long transcripts, reproduce on their longest transcripts when re-read on H200 GPUs? They feed the test of the probes' false alarms on honest runs of never-seen tasks, the Seed-OSS-36B and Qwen3-32B tool-output probe fits, and the find-fix-measure loop on simulator formatting.

## What we did

Re-read long transcripts from each dump's B200 shards on H200 and compared them with the stored activations: Seed-OSS-36B on one stage of the never-seen-task honest runs, on fully format-fixed simulated runs and on fully fixed runs with the file-view trailing-newline repair; and Qwen3-32B's dump of format-fixed runs used for its averaged tool-output probes.

## What we found

Three of four passed: Seed-OSS-36B on stage-7 fresh honest runs PASS, Seed-OSS-36B on fully format-repaired runs EXACT, Qwen3-32B on format-repaired runs for token-averaged tool-output probes PASS; the re-read of fully format-repaired runs with the file-view trailing-newline repair was cut by the billing watcher one run short (57 of 58) and wrote no comparison (NOTE_f9_stop_2026-10-14.md).

## In the post?

Not in the post.

## Status

Current (file-view trailing-newline repair part incomplete)

## Builds on

The 2026-10-13 full-fix and fresh-honest dumps

## Main scripts

- [`run_b200_longrepro.py`](run_b200_longrepro.py)

Result files the index names:

- [`b200long_fresh_s7_2026-10-14_140915.json`](results/b200long_fresh_s7_2026-10-14_140915.json)
- [`b200long_fixfull_2026-10-14_140915.json`](results/b200long_fixfull_2026-10-14_140915.json)
- [`b200long_fxavg_2026-10-14_140915.json`](results/b200long_fxavg_2026-10-14_140915.json)
- `b200long_fixfull_f9_2026-10-14_144352.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-14_q2_b200_longrepro/` at commit `d5200f9ff602edc325a5312d3fc6d0ceaa7c751d` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
