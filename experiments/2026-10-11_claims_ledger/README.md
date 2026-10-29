# 2026-10-11_claims_ledger

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does every result number on our internal status page's per-question answer lines come from a committed, script-written file, and does it match that file?

## What we did

A checker script maps each number on our internal status page (version 463) to a pinned commit, file and JSON key using claim-to-source specs, reads the value with git show and compares it at the page's rounding; \$0.

## What we found

README: 421 rows, 417 match, 3 mismatch, 1 readable only on S3, 0 without a source; 587 page numbers covered, 167 without a row (commit b25b72334).

## In the post?

Not in the post.

## Status

Current

## Builds on

Answer lines of our internal status page (inputs/answer_lines_v463.txt); result files across the folders of the first three research questions

## Main scripts

- [`run_claims_ledger.py`](run_claims_ledger.py)

Result files the index names:

- `claims_ledger_2026-10-11_034605.json` (not copied: strings over 300 characters)
- [`claims_ledger_2026-10-11_034605.md`](results/claims_ledger_2026-10-11_034605.md)
- `claims_ledger_2026-10-11_030830.json` (not copied: strings over 300 characters)
- [`claims_ledger_2026-10-11_030830.md`](results/claims_ledger_2026-10-11_030830.md)
- `s3_manifest_claims_ledger_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-11_claims_ledger/` at commit `46060302b7c211ce10995b347412f940d468ef0e` of our internal repository. 19 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
