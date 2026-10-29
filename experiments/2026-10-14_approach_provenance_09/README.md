# 2026-10-14_approach_provenance_09

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Did the simulated attack runs receive the same attack approach (stealth advice) and system prompt as real runs 1 and 2, task by task? And is GPT-5.6-Sol's side-task success more repeatable on tasks whose side task has an authored attack strategy?

## What we did

Modal CPU scan of the raw .eval logs of Qwen3.5-27B's real run 1, real run 2 and simulated attack runs on the 196-task pilot, comparing sha256 hashes of the attack approach and system message per task (no text read). A second part counted Sol's epoch-2 side-task success on the 196 tasks, split by whether the side task has a strategy file, with Wilson and Newcombe intervals and Fisher's exact test.

## What we found

Gate passed (193/192/191 mapped runs found); approach and system prompt identical across all three arms on 190 of 190 tasks (approach_hash_2026-10-14_125313.json, part_a.per_task.counts). Second part: epoch-2 success 136/178 (76.4%) with a strategy vs 10/18 (55.6%) without, difference +20.8 points (Newcombe 95% −0.2 to +43.4), Fisher p 0.084, underpowered (item2_t196_2026-10-14_130427.json).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (196-task file)

## Main scripts

- [`run_approach_hash.py`](run_approach_hash.py)
- [`run_item2_t196.py`](run_item2_t196.py)
- [`make_strategy_bits.py`](make_strategy_bits.py)

Result files the index names:

- `approach_hash_2026-10-14_124912.json` (not copied: result left out to keep the release under 120 MB)
- `approach_hash_2026-10-14_125313.json` (not copied: result left out to keep the release under 120 MB)
- `item2_t196_2026-10-14_130427.json` (not copied: result left out to keep the release under 120 MB)
- `strategy_bits_t196.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_approach_provenance_09/` at commit `89fc226f15206773b801a2dc8dfadedf1c02040b` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
