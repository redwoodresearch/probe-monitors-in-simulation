# 2026-10-09_q2_tpr_at_1pct

2026-10-09 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

What is the impactful-command probe's catch (true positive rate, TPR) at a threshold frozen at 1% false positive rate (FPR), real vs simulated, off-policy and on-policy?

## What we did

Reproduced each setup's reported numbers, then computed catch at a frozen 1% threshold with the realized false-positive rate per world, log-scale ROC curves and planted 5-point drops: off-policy (Qwen3.5-27B reading GPT-5.6-Sol, 530 tasks) and on-policy (Qwen3.5-27B acting, 20 tasks, September simulator).

## What we found

Off-policy catch 0.9081 real vs 0.9075 simulated (gap +0.06 points [-1.07, +1.34]), planted 5-point drop detected 200 of 200. On-policy pilot gap -1.89 points [-14.22, +9.45], a 5-point gap not ruled out.

## In the post?

Not in the post (early version of the catch-at-1% comparison; the post's numbers come from the records of the final pooled analysis on all 1,313 tasks).

## Status

Superseded by the 196-task scoring in [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) and then [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) (the final pooled analysis on all 1,313 tasks)

## Builds on

[2026-10-08_impactful_refit_bootstrap](../2026-10-08_impactful_refit_bootstrap/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`analyze_tpr_frozen.py`](analyze_tpr_frozen.py)
- [`plot_tpr_figs.py`](plot_tpr_figs.py)
- [`plot_offpolicy_2x2_catch.py`](plot_offpolicy_2x2_catch.py)

Result files the index names:

- `tpr_frozen_2026-10-10_110209.json` (not in this folder at the source commit)
- `tpr_frozen_2026-10-09_162427.json` (not in this folder at the source commit)
- `manifest_v2.json` (not copied: strings over 300 characters)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_q2_tpr_at_1pct/` at commit `da9eb789f4ac8212365f5a79d2188dbfabf03426` of our internal repository. 15 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
