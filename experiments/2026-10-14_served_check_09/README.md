# 2026-10-14_served_check_09

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Does scoring Qwen3.5-27B's runs in the context the agent was actually served (three variants: with its reasoning and the attack runs' own submit tool, with its reasoning and an honest submit tool, and without its reasoning), instead of the simplified layout the probes normally read, change the simulated-minus-real catch gap?

## What we did

Following the pre-registered rules, first checked that the usual layout reproduces version 6 of the "error to add" analysis (how many points a lab should add to its error bars when it measures probe catch in simulation), then computed per probe, readout and alarm level, and pooled, the change in the gap on the 90 pilot tasks held out for estimation, with 10,000 environment-stratified resamples; a fresh agent rechecked the record, and a post hoc variant ran on 2026-10-15.

## What we found

At reply readout 5%, pooled change −0.6 (95% −4.6 to +3.4) with reasoning retained and the original attack submit definition, −0.6 (−4.6 to +3.6) with reasoning retained and an honest submit definition, −0.3 (−1.9 to +1.4) with reasoning removed and an honest submit definition (all three with the registered donor system prompt), "no material change in the family's average signed gap"; at 1% "inconclusive at this n" (recheck/claims_table_212330.txt); independent recheck 9 of 9 checks match (recheck/RECHECK_212330.md).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_p2_replay_check](../2026-10-14_p2_replay_check/); [2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/) (v6)

## Main scripts

- [`run_served_read.py`](run_served_read.py)
- [`recheck/check_served_212330.py`](recheck/check_served_212330.py)
- [`recheck/make_claims_table.py`](recheck/make_claims_table.py)

Result files the index names:

- `served_read_real_2026-10-14_212330.json` (not copied: result file 1 MB or larger)
- `served_read_real_posthoc_2026-10-15_000854.json` (not copied: result file 1 MB or larger)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_served_check_09/` at commit `ec09ad9940735215004f8f5720f49a8e31a0e457` of our internal repository. 34 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
