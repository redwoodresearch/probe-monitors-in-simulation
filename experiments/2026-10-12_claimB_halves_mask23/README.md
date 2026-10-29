# 2026-10-12_claimB_halves_mask23

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does the halves gap (train on one half of the environments and test on the other, in both directions; simulated versus second-real flag rates at the 1% threshold) survive when numbers/IDs and whitespace are both masked in real and simulated runs?

## What we did

Re-read the evened halves runs with Qwen3-32B after masking numbers/IDs and whitespace in both worlds, refit the halves probes and scored simulated vs second-real runs at the frozen 1% alarm, with planted checks (the refit runner crashed once before any fit and was fixed).

## What we found

Codex fact-check: pooled gap fell to +0.32 points [-0.64, +1.73] (change -3.51 [-6.97, -1.29]), but one planted check failed (dev-to-test AUROC 0.893 below 0.910), so no reading.

## In the post?

Not in the post.

## Status

Exploratory; no reading (planted check failed); its two parts rerun separately in [2026-10-12_claimB_halves_mask_split](../2026-10-12_claimB_halves_mask_split/)

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/)

## Main scripts

- as-run scripts at c3793472 (on a side branch)
- the latest ref holds only PROVENANCE.md

Result files the index names:

- `halves_mask23_2026-10-12_120522.json` (not in this folder at the source commit)
- `codex_factcheck_halves_mask23_2026-10-12_120522.txt` (not in this folder at the source commit)

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_halves_mask23/` at commit `875f7b50bd376946ec83c5e1addd3dde05e91799` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
