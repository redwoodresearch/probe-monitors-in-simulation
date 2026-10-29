# 2026-10-08_q2_onpolicy_sizing

2026-10-08 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

How many tasks, and at what cost, would the full on-policy study need (60-125 tasks)?

## What we did

A local pricing script combining amendment 40's per-task precision with the (invalid) pilot's per-run costs.

## What we found

Planning only; about 74 tasks give a half-width of 0.03 on the low-alarm ranking score; the September simulator alone costs \$1,670 (60 tasks) to \$3,480 (125).

## In the post?

Not in the post.

## Status

Superseded by [2026-10-09_q2_september_sizing](../2026-10-09_q2_september_sizing/) (the sizing used for the launch)

## Builds on

[2026-10-07_phase2_probe_proposal](../2026-10-07_phase2_probe_proposal/) (amendment 40)

## Main scripts

- [`size_study.py`](size_study.py)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_q2_onpolicy_sizing/` at commit `3838105a7d49eaffa283e676a4ba8505f8646480` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
