# 2026-10-13_sce_width_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

What makes the 1% intervals of the simulated-control-evaluation probe comparisons so wide, and which remedies would narrow them?

## What we did

From stored scores (Qwen3-32B reading GPT-5.6-Sol runs), split each target's 1% interval into task sampling and threshold noise, measured nested-model spread, and priced remedies (Harrell-Davis thresholds, pooled honest calibration, same-model pools, partial AUROC, more honest runs). \$0, local.

## What we found

sce_width_v2_POSTHOC_2026-10-13_153829.printout.txt: for D (second-real-run catch under the repaired simulation-only probe minus catch under the real-trained probe) at 1% the width is 21.1 points, 98% of it from threshold noise (task-only width 4.2); for the simulated-minus-real gap at 1% the width is 5.8, 98% from task sampling; no tested remedy makes D at 1% pass; the 5% versions pass. Diagnostic only.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (a side branch @ a74cbca7)

## Main scripts

- [`analyze_sce_width_POSTHOC.py`](analyze_sce_width_POSTHOC.py)
- [`analyze_sce_width_a3_POSTHOC.py`](analyze_sce_width_a3_POSTHOC.py)
- [`analyze_sce_width_scaling_POSTHOC.py`](analyze_sce_width_scaling_POSTHOC.py)

Result files the index names:

- [`sce_width_POSTHOC_2026-10-13_152755.printout.txt`](results/sce_width_POSTHOC_2026-10-13_152755.printout.txt)
- [`sce_width_v2_POSTHOC_2026-10-13_153829.printout.txt`](results/sce_width_v2_POSTHOC_2026-10-13_153829.printout.txt)
- [`sce_width_a3_POSTHOC_2026-10-13_153245.printout.txt`](results/sce_width_a3_POSTHOC_2026-10-13_153245.printout.txt)
- [`sce_width_scaling_POSTHOC_2026-10-13_153521.printout.txt`](results/sce_width_scaling_POSTHOC_2026-10-13_153521.printout.txt)

## Files here

3 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-13_sce_width_POSTHOC/` at commit `2bc8256bf45258a17ac7fac581d545d117ce42cb` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
