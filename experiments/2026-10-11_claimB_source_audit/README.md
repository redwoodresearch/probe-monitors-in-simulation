# 2026-10-11_claimB_source_audit

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Was any simulated arm in the fixed-simulator test compared with the very real run it was built from (which would make the two look alike for free), and does the clock rule (which shifts simulated timestamps to match real runs) leave changes that time masking misses?

## What we did

Read-only audit citing file, line and commit for every real-versus-simulated result of that test; follow-ups counted the clock rule's leftover edits after time masking (one Modal CPU job) and each September run's start metadata.

## What we found

Commit 1429d57ba: the k = 1 test and the Oct 7 matched-step pairs flagged, rows with no text cleanup partly affected (clock from real run 1), everything else OK. FOLLOWUPS.md: 176 of 2,167 clock edits survive time-and-date masking, in 102 of 939 simulated runs (10.9%).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-11_claimB_k_replies](../2026-10-11_claimB_k_replies/)

## Main scripts

- [`analyze_POSTHOC_r6_level1_modal.py`](analyze_POSTHOC_r6_level1_modal.py)
- [`analyze_POSTHOC_start_metadata.py`](analyze_POSTHOC_start_metadata.py)

Result files the index names:

- `r6_level1_POSTHOC_2026-10-11_210109.json` (not copied: result left out to keep the release under 120 MB)
- `start_metadata_POSTHOC_2026-10-11_210252.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_source_audit/` at commit `d663b81b92df25bf44aebb2d5d89f5566515a725` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
