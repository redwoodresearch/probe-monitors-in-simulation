# 2026-10-15_solr2_audit

2026-10-15 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Did any record read the superseded off-policy probe score folders for two quarters of the +311 added tasks?

## What we did

Mapped 100 sha256 values of old and reissued score files and scanned records for them.

## What we found

No audited record read a superseded /sol/ folder of the second or third quarter (result JSON).

## In the post?

Not in the post.

## Status

Current

## Builds on

2026-10-14 off-policy probe scoring

## Main scripts

- [`mkfilemap.py`](mkfilemap.py)
- [`scan3.py`](scan3.py)

Result files the index names:

- `solr2_audit_2026-10-15.json` (not copied: strings over 300 characters)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_solr2_audit/` at commit `8d6a35191d80466855e4b52b17b553d45bb4aad6` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
