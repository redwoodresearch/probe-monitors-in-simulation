# 2026-10-14_paired_firstcut

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

For probes trained in many ways, does catch on Qwen3.5-27B's real attack runs track catch on its simulated attack runs, task by task, with tight paired intervals? This was an independent first answer to the supervisor's top-priority question.

## What we did

From stored per-run scores of the on-policy probe, the off-policy probe trained on GPT-5.6-Sol's runs (averaged over tool outputs, and at commands), two probes trained on format-fixed simulated runs, a positive-control probe and off-policy probes adapted with 10 or 30 of Qwen3.5-27B's own runs, it set one threshold per probe from two pooled batches of real honest runs (A and B) and computed simulated-minus-second-real and second-minus-first-real differences with paired intervals and McNemar tests on 177–183 pilot tasks. It then compared these with the other first cut ([2026-10-14_priority1_firstcut](../2026-10-14_priority1_firstcut/)).

## What we found

Agreed with an independent implementation on all 80 shared keys (0 disagreements). On-policy probe sim − run 2 at 1%: +0.0 points (95% −3.4 to +3.4, 177 tasks); positive control +30.5 (95% +22.5 to +38.5) (compare and paired_firstcut printouts).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_priority1_firstcut](../2026-10-14_priority1_firstcut/); [2026-10-14_headline_recheck_09](../2026-10-14_headline_recheck_09/) (score files)

## Main scripts

- [`run_paired_firstcut.py`](run_paired_firstcut.py)
- [`paired_stats.py`](paired_stats.py)
- [`check_compare_2e.py`](check_compare_2e.py)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_paired_firstcut/` at commit `b51860b017348a5c750480708d6aedb86716534c` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
