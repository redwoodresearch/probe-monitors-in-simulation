# 2026-10-16_supersede_diag

2026-10-16 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Why did marking the re-extracted attack folders of the paper's remaining 493 tasks as replacing (superseding) the old folders fail, when their activations should be identical?

## What we did

Compared the rows shared between the old and new score folders (keys and magnitudes only) and wrote a rule that allows tiny numerical score differences when marking folders as superseded.

## What we found

Activations byte-identical on all 482 shared attack epochs at layer 63 (all 64 layers checked for two keys); scores differ by at most 1.78e-14 (scorer row grouping), ruled float noise (RULE_amendment_49a.md).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-15_q3_sp493](../2026-10-15_q3_sp493/)

## Main scripts

- [`run_supersede_diag.py`](run_supersede_diag.py)
- [`supersede_sp493_v2.py`](supersede_sp493_v2.py)

Result files the index names:

- `supersede_diag_2026-10-16_032201.json` (not copied: strings over 300 characters)
- `supersede_diag_l63_2026-10-16_032848.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_supersede_diag/` at commit `9d09c34f1f88395985a1f10660246ce40b9122e1` of our internal repository. 16 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
