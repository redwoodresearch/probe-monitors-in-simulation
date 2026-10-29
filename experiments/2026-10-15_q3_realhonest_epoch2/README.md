# 2026-10-15_q3_realhonest_epoch2

2026-10-15 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Can a second real honest run per task tighten the real alarm threshold?

## What we did

Launched a second independent real honest run per pooled task in three real-only launches (one per origin) on Modal H200 vLLM servers.

## What we found

Total \$384.12 (Modal GPU \$320.53 billed) (spend row 2026-10-15 17:57); target pool 1,685 to up to 3,393 runs (per the experiment's written rules).

## In the post?

In the post: indirect: real honest pool for the real threshold (Appendix A, Figure 1)

## Status

Current

## Builds on

[2026-10-14_q3_new_mains_launch](../2026-10-14_q3_new_mains_launch/) quote

## Main scripts

- [`make_tasks_ep2_t196.py`](make_tasks_ep2_t196.py)
- [`launch_ep2_t196.sh`](launch_ep2_t196.sh)
- [`launch_ep2_311.sh`](launch_ep2_311.sh)
- [`launch_ep2_nm.sh`](launch_ep2_nm.sh)
- [`gate_ep2.py`](gate_ep2.py)
- [`split_311.py`](split_311.py)

Result files the index names:

- `end_t196_2026-10-15.json` (not copied: result left out to keep the release under 120 MB)
- `end_311_2026-10-15.json` (not copied: result left out to keep the release under 120 MB)
- `end_nm_2026-10-15.json` (not copied: result left out to keep the release under 120 MB)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

30 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_q3_realhonest_epoch2/` at commit `443650eb5aef0f83796c9223c7592a066a6496e9` of our internal repository. 18 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
