# 2026-10-20_appI_counts

2026-10-20 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

How many real honest and real attack runs errored, by environment, for the list of left-out runs?

## What we did

Counted errored real-side runs (honest pools and attack runs 1 and 2) from committed records with git show and jq, pinned by sha256; two versions.

## What we found

The post's Appendix I: at least 39 real honest runs left out (26 model-registry, 10 slack-clone, 3 tiktok); real attack runs with two empty replies were 11 of 2,744.

## In the post?

In the post: Appendix I (honest runs left out of the thresholds; errored real runs)

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) records

## Main scripts

- [`count_appI_real.zsh`](count_appI_real.zsh)

Result files the index names:

- `appI_real_counts_2026-10-20_021136.json` (not copied: strings over 300 characters)
- `appI_real_counts_2026-10-20_022807.json` (not copied: strings over 300 characters)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/7e0abf8c3/count_appI_real.zsh`](as_run/7e0abf8c3/count_appI_real.zsh) from commit `7e0abf8c3`: results/appI_real_counts_2026-10-20_021136.json records commit 7e0abf8c3, where this script had this version. The later appI_real_counts_2026-10-20_022807.json records commit b86d7b12f, where the script matches the copy in this folder.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 2 result files the index names, they match in 1 script-and-commit pair, differ in 1. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_appI_counts/` at commit `c82df6d2c7b140192532248d4432aa1c8c3901ae` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 1 file under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
