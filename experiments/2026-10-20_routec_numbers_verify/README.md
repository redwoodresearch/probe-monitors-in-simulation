# 2026-10-20_routec_numbers_verify

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does every number printed in the draft post match its source record?

## What we did

Built a provenance map (printed number to record key path and comparison rule) and a verifier that checks each printed number, with constants sources, tolerance tables and repeated passes (earlier drafts of the post).

## What we found

v17 pass: 251 of 251 map rows pass (232 equal, 6 preliminary, 13 not on the final pipeline); 761 printed numbers partitioned, 11 residual items stopped for follow-up (NOTE_numv9).

## In the post?

Not in the post (verifies printed numbers across the post).

## Status

Current

## Builds on

[2026-10-17_routec_slot_readout](../2026-10-17_routec_slot_readout/); [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/)

## Main scripts

- [`run_numbers_verify.py`](run_numbers_verify.py)
- [`verify_core.py`](verify_core.py)
- [`verify_post.py`](verify_post.py)
- [`predicates.py`](predicates.py)

Result files the index names:

- `provenance_v84_final17_0141.jsonl` (not copied: per-run records (.jsonl))
- `nonresult_list_2026-10-20_045432.json` (not copied: result left out to keep the release under 120 MB)

## Files here

39 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_numbers_verify/` at commit `b03d699962af975fec4fae811a94a39dcf7cc133` of our internal repository. 196 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
