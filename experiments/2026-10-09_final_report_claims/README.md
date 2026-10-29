# 2026-10-09_final_report_claims

2026-10-09 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Do failed honest runs still claim success in their final message, and are there enough such runs to label a deceptive-final-report probe?

## What we did

Script-rule count (no judge) over Qwen3.5-27B's strict-rerun honest runs: classifies each run's final visible message as claim, hedged or no summary and crosses it with the real scorer's verdict.

## What we found

With a real verdict, 7 of 13 (54%) failed Qwen3.5-27B runs claim success vs 22 of 23 (96%) passed runs; 7 labelled runs suffice for a small test set, not for training.

## In the post?

Not in the post.

## Status

Current (label count; used by [2026-10-10_q2_false_claims](../2026-10-10_q2_false_claims/))

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (strict rerun)

## Main scripts

- [`count_claims.py`](count_claims.py)

Result files the index names:

- [`claims_2026-10-09_181007.json`](results/claims_2026-10-09_181007.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-09_final_report_claims/` at commit `e7f057eb26690a73887afc844d70f4b64f7c6ec5` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
