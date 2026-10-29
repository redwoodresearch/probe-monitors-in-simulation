# 2026-10-20_routec_rerun_honest_all

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Independently re-derive the honest-run detector numbers in Table 7 (rows 4-5).

## What we did

Fresh implementer refit the registered recipe independently and replayed stored weights per fold (hash-pinned shards, then replay), with toy controls.

## What we found

Independent refit: four fit verdicts UNRESOLVED as pre-stated, rows stay PRELIMINARY (NOTE_results); replay phase gates passed (NOTE_routeA_results).

## In the post?

Not in the post (verifies Table 7 rows 4-5, Appendix E).

## Status

Current

## Builds on

[2026-10-16_honest_sim_detector_all](../2026-10-16_honest_sim_detector_all/)

## Main scripts

- [`check_honest_all.py`](check_honest_all.py)

Result files the index names:

- `honest_all_rerun_2026-10-20_152036.json` (not copied: result left out to keep the release under 120 MB)
- `honest_all_replay_2026-10-20_174318.json` (not copied: result left out to keep the release under 120 MB)

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_rerun_honest_all/` at commit `e694627ab5af6e1b2f05ac783177576480624d2a` of our internal repository. 41 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
