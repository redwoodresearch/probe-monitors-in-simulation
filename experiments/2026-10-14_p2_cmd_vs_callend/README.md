# 2026-10-14_p2_cmd_vs_callend

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

How often do the "command-end" token (last token inside the command parameter, read by the published probes) and the content-only "end-of-call" token (last content token of the call, read by the Sol-trained probe) land on the same token?

## What we did

Scores-free count on Modal CPU over every staged call of the 196-task corrected-render inputs (all six worlds), split by tool type, comparing the two readout positions.

## What we found

About 87% of calls with both readouts share the same token (first real honest arm 5,480 of 6,280; second 5,434 of 6,204); they differ mainly on file edits, where the command end comes before the end of the call (cmd_vs_callend printout).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/) (Amendment 9)

## Main scripts

- [`run_cmd_vs_callend.py`](run_cmd_vs_callend.py)

Result files the index names:

- [`cmd_vs_callend_2026-10-14_190758.json`](results/cmd_vs_callend_2026-10-14_190758.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_p2_cmd_vs_callend/` at commit `81c2974e1d1a479c699e5d1807548e5dae9d4d9d` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
