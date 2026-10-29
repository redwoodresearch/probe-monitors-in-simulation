# 2026-10-15_cut_sensitivity

2026-10-15 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

Does reading only the first 32,768 tokens of each run bias the comparison, and what is needed to re-extract every run at full length?

## What we did

Counted the runs truncated by the 32,768-token cut, checked how the gap changes on length-matched runs, priced a full-length re-extraction, and built the run lists used to re-prepare every group of runs at full length (Qwen3.5-27B's pilot runs in two fix versions, the interleaved launch, the paper's remaining 493 tasks and the second batch of real honest runs); the second fix version of the pilot list was later rebuilt through a gated wrapper script.

## What we found

Length-matching moved the registered gap by under 2 points for 9 of 9 reply probes at per-run 5% (max 1.5); full re-extraction quoted at about \$323 raw / \$420 with 30% for the common comparison (README).

## In the post?

In the post: indirect: the full-length run inventories pinned by the final records behind Figure 1 and Tables 5/8

## Status

Current (inventories used by the final full-length records)

## Builds on

[2026-10-14_q3_overnight_chain](../2026-10-14_q3_overnight_chain/); [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) config

## Main scripts

- [`run_cut_census.py`](run_cut_census.py)
- [`run_cut_sens455.py`](run_cut_sens455.py)
- [`make_inventory_b_fix_v2.py`](make_inventory_b_fix_v2.py)
- [`make_il_scope_v4.py`](make_il_scope_v4.py)
- [`analyze_cutsens.py`](analyze_cutsens.py)

Result files the index names:

- `cut_census_2026-10-15_120553.json` (not in this folder at the source commit)
- `cutsens455_2026-10-15_121722.json` (not in this folder at the source commit)
- `quote_fulllength_2026-10-15.json` (not in this folder at the source commit)
- `inventory_b_fix_v2_2026-10-17_081456.json` (not copied: result file 1 MB or larger)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_cut_sensitivity/` at commit `fe8ba4599bc86016087858b523b45981f0a2ae44` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 5 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
