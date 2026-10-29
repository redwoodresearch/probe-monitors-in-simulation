# 2026-10-14_fpr_sweep_xcheck

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does an independent implementation of the false-alarm-rate sweep scored per unit (each tool output or command), written from the analysis plan without seeing the original code, reproduce the original numbers?

## What we did

An independent agent built its own sweep from the per-row score files (the probe trained on GPT-5.6-Sol's runs, tool-output and command units, 183-task pilot) and the analysis-plan text, then compared it with the original results by key, flagging differences over 0.5 points.

## What we found

348 keys compared; point values identical (max diff 0), 31 keys differ only at interval endpoints, by about one task step (compare_2e_2026-10-14_131359.printout.txt).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_priority1_spec_09](../2026-10-14_priority1_spec_09/) (spec); another working session's corrected-render sweep

## Main scripts

- [`run_sweep_xcheck.py`](run_sweep_xcheck.py)
- [`check_compare_2e.py`](check_compare_2e.py)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_fpr_sweep_xcheck/` at commit `d6cf94b691a6642c4a78d612962d3a111eb945d7` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
