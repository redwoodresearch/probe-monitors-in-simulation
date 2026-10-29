# 2026-10-11_target_touch_labels

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Which of Qwen3.5-27B's commands touch the side task's target, in real attack runs 1 and 2, simulated attack runs and honest runs (labels for a planned monitor of target-touching commands and for the measure of how often attack runs touch the target)?

## What we did

Rule-based word-matching labels of every call against each side task's target inventory (version 1, repaired version 2, and version 3 adding a flag for targets shared with the main task), run locally outside Claude, with synthetic test cases and hash-only inventories in git.

## What we found

README: version 2 positives 2,734 of 13,448 calls (real run 1), 2,549 (run 2), 2,600 (simulated), 1,662 and 1,685 (the two real honest arms); lexical mentions only; version 2 pending a fresh blind audit; version 3 is the current handoff.

## In the post?

Not in the post.

## Status

Current (v3)

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (196-task runs)

## Main scripts

- [`analyze_target_labels.py`](analyze_target_labels.py)
- [`analyze_target_inventory_hashes.py`](analyze_target_inventory_hashes.py)
- [`analyze_shared_main_targets.py`](analyze_shared_main_targets.py)
- [`analyze_shared_main_validation.py`](analyze_shared_main_validation.py)

Result files the index names:

- [`artifact_manifest.json`](results/v2/artifact_manifest.json)

## Files here

7 code and configuration files and 6 result files, copied byte for byte from `experiments/2026-10-11_target_touch_labels/` at commit `fcaff58f86d0415d9509673b1251afc8e5d403a0` of our internal repository. 16 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
