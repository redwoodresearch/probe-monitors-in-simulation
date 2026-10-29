# 2026-10-15_epoch2_poolkeys

2026-10-15 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Which second-draw real honest runs join which honest pool used to set real thresholds?

## What we did

Built by-key lists of epoch-2 real honest runs admitted to each epoch-1 pool (replicates only), from manifests and .eval hashes; no scores read; plus a list of missing replicates.

## What we found

Pool-key artifact produced in four revisions; no values.

## In the post?

In the post: indirect: real-threshold honest pools (epochs 1 and 2) behind Figure 1 and Appendix A

## Status

Current

## Builds on

[2026-10-15_q3_realhonest_epoch2](../2026-10-15_q3_realhonest_epoch2/)

## Main scripts

- [`make_epoch2_poolkeys.py`](make_epoch2_poolkeys.py)
- [`list_missing_replicates.py`](list_missing_replicates.py)

Result files the index names:

- `epoch2_poolkeys_2026-10-15_214512.json` (not copied: result file 1 MB or larger)
- `missing_replicates_2026-10-15_215604.json` (not copied: strings over 300 characters)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 2 result files the index names, they match in 3 script-and-commit pairs, and in 1 the script was not in the folder at that commit. Other result files, from earlier or side runs, give 0 script-and-commit pairs that match, 4 that differ and 4 where the script was not in the folder; we did not copy those versions.

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_epoch2_poolkeys/` at commit `0282a37563ca8f8528f8a5a9b3cce527a86529a2` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
