# 2026-10-15_fibs_reaggregate

2026-10-15 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does averaging the deception probe over a run dilute a signal that sits in a few steps?

## What we did

Re-aggregated saved per-row deception-probe scores with other run rules (max, top-2, top-3, last 20%, any step) on the 183-task pilot; plus a fidelity note comparing our inputs with the probe authors' template.

## What we found

Any-step rule at 1% on commands: real catch 0.685, sim minus real +0.030 [-0.036, +0.094]; mean rule had caught 9.7% (printout; the experiment's written rules). The post later scores every action and alarms on any.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

2026-10-14 deception-probe score record

## Main scripts

- [`run_fibs_reaggregate.py`](run_fibs_reaggregate.py)

Result files the index names:

- `fibs_reagg_2026-10-15_113213.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_fibs_reaggregate/` at commit `cad588d2a9dbbdfc54390e4feddec68d09fe442f` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
