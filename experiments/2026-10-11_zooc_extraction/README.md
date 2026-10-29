# 2026-10-11_zooc_extraction

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Extract Qwen3.5-27B activations of the new real GPT-5.6-Sol honest runs bought for the credential-monitor comparison, with the same pipeline as the earlier version-4 all-layers activation dump.

## What we did

Staged the new runs (the batch of 11 October) as version-4-format reader inputs, checked the reader against version 4, and extracted activations at the end of each tool call and each tool output at all layers on Modal H200 GPUs, stage by stage (stages 1 to 6); the data are sealed as an unused confirmation set, so this folder only extracts and checks.

## What we found

Commits: stage 6 extracted, gates passed, \$1.60 billed for that stage (e26336534); no analysis here (sealed).

## In the post?

Not in the post.

## Status

Current (sealed)

## Builds on

[2026-10-11_q2_new_real_runs](../2026-10-11_q2_new_real_runs/); [2026-10-06_qwen35_replication](../2026-10-06_qwen35_replication/) (version-4 extractor)

## Main scripts

- [`stage_zooc.py`](stage_zooc.py)
- [`check_reader_vs_v4.py`](check_reader_vs_v4.py)
- [`spawn_extract.py`](spawn_extract.py)
- [`check_extract_gates.py`](check_extract_gates.py)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_zooc_extraction/` at commit `26304604e6744ec3d68bc2bfa31c42db0d545515` of our internal repository. 34 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
