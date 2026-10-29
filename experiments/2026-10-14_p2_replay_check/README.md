# 2026-10-14_p2_replay_check

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Served-context sensitivity: does reading Qwen3.5-27B's runs in the exact token layout the policy saw (the offered-tool list, the attack runs' own submit tool, the policy's reasoning) change the probes' scores?

## What we did

Priced, staged, extracted (41 H200 shards for three served-context layouts: as served with the honest system prompt swapped in, the same with the honest submit tool too, and that with reasoning emptied) and scored the frozen probes on these layouts of the pilot runs; the latest ref holds only the pricing rule and a task subset, while the staging, extraction and scoring records are on a side branch.

## What we found

Scoring completed 10-14 20:55 with its gate 20 of 20 byte-identical and 93,139 units assigned (commit aa8ad8044 message); the reading is in [2026-10-14_served_check_09](../2026-10-14_served_check_09/).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_p2_dt_layout_diff](../2026-10-14_p2_dt_layout_diff/); [2026-10-14_priority1_spec_09](../2026-10-14_priority1_spec_09/) (its appended rule for reading the served-context sensitivity)

## Main scripts

- [`extract_served.py`](extract_served.py)
- [`inspect_served.py`](inspect_served.py) (on a side branch)

Result files the index names:

- `stage_served_2026-10-14_200651.json` (not in this folder at the source commit)
- `score_served_manifest_2026-10-14_204921.slim.json` (not in this folder at the source commit)

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_replay_check/` at commit `8d9fa37403d5cb53ad3f70f876bf065723bb3ad3` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 2 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
