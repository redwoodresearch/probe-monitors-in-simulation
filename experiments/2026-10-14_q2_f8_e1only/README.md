# 2026-10-14_q2_f8_e1only

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

The fix for simulator error-message templates was derived from both real runs, but gaps are measured against real run 2. Does deriving it from real run 1 only change anything?

## What we did

Copied the as-run staging code, passed a reproduction gate against the as-run stage records, then re-derived that fix's templates from real run 1's errored outputs only and counted changed templates, runs and tool outputs per arm (earlier phase, GPT-5.6-Sol runs).

## What we found

Gates passed; 5 of the 14 as-run templates drop out (9 remain, no rule changes), changing 11 runs and 11 tool outputs across all arms (f8_e1only_POSTHOC_2026-10-14_153206.json).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/)

## Main scripts

- [`run_f8_e1only.py`](run_f8_e1only.py)

Result files the index names:

- [`f8_e1only_POSTHOC_2026-10-14_153206.json`](results/f8_e1only_POSTHOC_2026-10-14_153206.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_q2_f8_e1only/` at commit `ea30623c6566fff91913d171026385c627761521` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
