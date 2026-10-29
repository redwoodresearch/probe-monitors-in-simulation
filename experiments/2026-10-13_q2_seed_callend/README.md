# 2026-10-13_q2_seed_callend

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Does the finding that the mixed recipe's overstatement (probes trained on simulated attacks vs real honest runs) nearly vanishes when the probe reads the agent's commands with Qwen3-32B replicate on a second reader, Seed-OSS-36B?

## What we did

Built a new Seed-OSS-36B transcript rendering that keeps a readout at the end of each tool call and re-extracted all seven layers for 5,701 runs (real honest, first and second real attack, simulated attack and honest, and simulated runs with all formatting repairs). The automatic cost watchers stopped the first launch during a laptop network outage at 1,732 runs (\$78); a resume completed all 5,701 (567,126 rows, final \$231.31 per the commit log).

## What we found

Only launch and stop records are committed at the latest ref; commit 48baf8455 (10-13 22:22) records the fanned-out check: PASS, 5,701 of 5,701 transcripts, 0 non-finite, reproduction bit-exact. No probe result is in this folder: the dump was analyzed in [2026-10-13_why_steps](../2026-10-13_why_steps/) (why_seed_run_2026-10-13_221350.json), where its readings are marked SUSPECT because 41 of 128 shards ran on B200 GPUs without a long-context re-read (NOTE_SUSPECT_B200_seed.md); its extraction check is also cited by [2026-10-13_q3_layer_stability](../2026-10-13_q3_layer_stability/) and [2026-10-13_q3_loeo](../2026-10-13_q3_loeo/).

## In the post?

Not in the post.

## Status

Exploratory (extraction done; readings built on it marked SUSPECT)

## Builds on

[2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/) (arms and extractor); [2026-10-13_why_steps](../2026-10-13_why_steps/) (the Qwen3-32B command-readout finding)

## Main scripts

- [`check_seedcall.py`](check_seedcall.py)
- [`check_seedcall_fan.py`](check_seedcall_fan.py) (extractor run_extract_seedcall.py on S3)

Result files the index names:

- [`extract_launch_2026-10-13_seedcall_seed-oss-36b_v1_2026-10-13_175757.json`](results/extract_launch_2026-10-13_seedcall_seed-oss-36b_v1_2026-10-13_175757.json)
- `extract_stop_record_2026-10-13_182330.json` (not copied: strings over 300 characters)
- `s3_manifest_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_q2_seed_callend/` at commit `4aa46c6eb740927ea58ec871cf674e9d4c0a62b5` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
