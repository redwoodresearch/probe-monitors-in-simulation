# 2026-10-11_label_artifacts

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Archive the frozen target-touch and credential screening labels from the pull request that added them so later analyses can fetch and verify them.

## What we did

Uploaded the labels, per-run counts, source hashes and scorer target inventory to the private S3 bucket with a hash manifest; a fetch command verifies every object.

## What we found

README: 39 objects uploaded and verified byte for byte; no transcript text or credential values in git.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_target_touch_labels](../2026-10-11_target_touch_labels/); [2026-10-11_credential_count](../2026-10-11_credential_count/)

## Main scripts

- probe_shift.label_artifacts (module in src)

Result files the index names:

- `legacy_v1_manifest.json` (not copied: JSON outside results/ and config/)

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_label_artifacts/` at commit `fcaff58f86d0415d9509673b1251afc8e5d403a0` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
