# 2026-10-11_claimB_attachment_census

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Do the staged inputs behind the fixed-simulator test (whether a probe can tell runs from the formatting-fixed simulator from real runs) and their source logs contain unresolved Inspect "attachment://" placeholders instead of text?

## What we did

One Modal CPU container counted placeholders per set, arm, role and field in that test's four staged input sets (formatting-fixed data version 2; raw and time-masked text; inputs for the main probe and for the every-token readout variant); a laptop script counted them in 60 seeded source logs (first and second real attack runs, the repaired version-4 real attack runs, and GPT-6-Astra simulated attack and honest runs). Counts only.

## What we found

Commit messages 85cf2452b and 54457ee09: 0 placeholders in all four claim-B staged sets and 0 in the sample messages of the 60 source logs; placeholders occur only in events (1,935 to 11,849 per 10 logs).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (fixed simulator v2 probe inputs)

## Main scripts

- [`run_census_modal.py`](run_census_modal.py)
- [`analyze_source_logs.py`](analyze_source_logs.py)

Result files the index names:

- [`attachment_census_2026-10-11_165018.json`](results/attachment_census_2026-10-11_165018.json)
- `source_log_census_2026-10-11_165504.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_claimB_attachment_census/` at commit `54457ee0972ceb5055502066cf42223003497288` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
