# 2026-10-13_robust_1pct_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Can the 1% false-alarm intervals be made more robust, and do the committed intervals actually reach 95% coverage?

## What we did

Built a shared module (src/probe_shift/tail_threshold.py) with order-statistic, Harrell-Davis and Gaussian thresholds, pooled calibration, and percentile, smoothed and order-statistic-band intervals; reproduced the committed values, then measured coverage by simulation for the simulated-control-evaluation probe targets (Qwen3-32B reading GPT-5.6-Sol), Seed-OSS-36B's gap and the matched-false-alarm gap. Modal CPU, about \$0.3.

## What we found

robust_sce_v2_POSTHOC_2026-10-13_172216.printout.txt: the committed order-statistic percentile interval covers 93 to 97% for gaps but only 76% for one catch target at 1%; the smoothed bootstrap covers 96 to 99% at 1%; Gaussian thresholds under-cover (46% to 86% at 1%). Diagnostic only; committed verdicts stand.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_sce_width_POSTHOC](../2026-10-13_sce_width_POSTHOC/) (inputs); [2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/); [2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/)

## Main scripts

- [`run_robust_sce_modal.py`](run_robust_sce_modal.py)
- [`analyze_robust_sce_POSTHOC.py`](analyze_robust_sce_POSTHOC.py)
- [`run_robust_g3_modal.py`](run_robust_g3_modal.py)
- [`analyze_robust_g3_POSTHOC.py`](analyze_robust_g3_POSTHOC.py)
- [`run_robust_matched_modal.py`](run_robust_matched_modal.py)
- [`analyze_robust_matched_POSTHOC.py`](analyze_robust_matched_POSTHOC.py)
- [`sizing_matched_decomp_POSTHOC.py`](sizing_matched_decomp_POSTHOC.py)
- [`sizing_matched_pooled_POSTHOC.py`](sizing_matched_pooled_POSTHOC.py)

Result files the index names:

- [`robust_sce_v2_POSTHOC_2026-10-13_172216.printout.txt`](results/robust_sce_v2_POSTHOC_2026-10-13_172216.printout.txt)
- `robust_g3_v3_POSTHOC_2026-10-13_173242.printout.txt` (not copied: lines over 300 characters)
- `robust_matched_v2_POSTHOC_2026-10-13_173242.printout.txt` (not copied: lines over 300 characters)
- [`sizing_matched_pooled_POSTHOC_2026-10-13_170803.printout.txt`](results/sizing_matched_pooled_POSTHOC_2026-10-13_170803.printout.txt)

## Files here

11 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_robust_1pct_POSTHOC/` at commit `1068cd300ea07b0b436094e165b994cbc1a693a4` of our internal repository. 18 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
