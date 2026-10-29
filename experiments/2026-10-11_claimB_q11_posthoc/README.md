# 2026-10-11_claimB_q11_posthoc

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Checks asked for by a panel of AI skeptic reviewers on why the realism probe beats text classifiers after the simulator's formatting fixes: does the main probe's own 1% catch beat a selection-aware shuffle null, and does its ranking-score (AUROC) edge over the TF-IDF text classifier hold with task-resampled intervals?

## What we did

\$0 post hoc on stored data: redid the main probe's layer and regularization selection on all 100 cell-grouped label-shuffle columns of its grid to get a null for catch at 1% and 5% false alarms, and paired the probe's run scores with the TF-IDF classifier's on the same 946 test runs.

## What we found

PLAN.md results: the main Qwen3-32B end-of-output realism probe caught 2.75% at 1% vs a null 95th percentile of 2.96%, p = 0.139 (not above its null); at 5% 40.0% vs 7.6%; probe minus TF-IDF +0.172 [+0.139, +0.205], but two real runs also give +0.040 [+0.003, +0.077].

## In the post?

Not in the post.

## Status

Current (post hoc)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (its first probe step's grid and TF-IDF scores); [2026-10-11_claimB_catch_null](../2026-10-11_claimB_catch_null/)

## Main scripts

- [`analyze_POSTHOC_stepA_catch_null.py`](analyze_POSTHOC_stepA_catch_null.py)
- [`analyze_POSTHOC_probe_vs_tfidf.py`](analyze_POSTHOC_probe_vs_tfidf.py)
- [`plot_POSTHOC_q11.py`](plot_POSTHOC_q11.py)

Result files the index names:

- `stepA_catch_null_POSTHOC_2026-10-11_040431.json` (not copied: result left out to keep the release under 120 MB)
- `stepA_catch_null_POSTHOC_2026-10-11_040431.png` (not copied: result file type not copied (.png))
- `probe_vs_tfidf_POSTHOC_2026-10-11_040340.json` (not copied: result left out to keep the release under 120 MB)
- `probe_vs_tfidf_POSTHOC_2026-10-11_040340.png` (not copied: result file type not copied (.png))

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_q11_posthoc/` at commit `35e410416b90e30697aac6fd1920da7ec17c5c41` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
