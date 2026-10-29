# 2026-10-14_runlength_check

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Is the roughly +3-point simulated-minus-real catch gap of the end-of-call (command) probe caused by simulated runs issuing more commands, giving "any command crosses" more chances?

## What we did

Counts-only scan of the raw .eval logs joined to the Sol-trained probe's per-command scores on the 183-task pilot: commands per run, the per-run share of flagged commands, and whether extra simulated catches sit in tasks where the real run hit tool errors, with task bootstraps.

## What we found

Simulated runs had +2.05 more scored commands per run (95% +0.18 to +3.91), but the share of flagged commands showed no gap (+0.23 points, 95% −0.46 to +0.93 at 1% unit level) while catch was +3.01 (95% −3.24 to +9.32), pointing to run length rather than per-command scoring (runlength_2026-10-14_140115.printout.txt).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_fpr_sweep_xcheck](../2026-10-14_fpr_sweep_xcheck/); [2026-10-14_tool_schema_behaviour_09](../2026-10-14_tool_schema_behaviour_09/)

## Main scripts

- [`run_runlength.py`](run_runlength.py)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_runlength_check/` at commit `ac8257127fbfc0ba7af135ac7134df737e560697` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
