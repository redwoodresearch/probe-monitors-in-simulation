# 2026-10-13_countmatched_coverage_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Do the 1% confidence intervals behind the 'simulation-trained detector matches a real-trained detector at equal training count' comparisons actually cover the truth, and how many honest runs would a reliable interval need?

## What we did

Extended the robust-1% coverage study to the count-matched comparisons (second real run's catch under a simulation-trained cell minus a count-matched real-trained detector, averaged over 5 subsample draws; Qwen3-32B reading GPT-5.6-Sol, plus Seed-OSS-36B): reproduced the stored values, then simulated coverage of fixed-draw and resampled-draw intervals in a smoothed synthetic world, and projected interval width against honest-pool size. Modal CPU, about \$0.5.

## What we found

Result JSONs are on S3 (manifest_cmcov_results_v1.json). Codex round 5 (agent summary checked against the JSON): with the current honest pool the projected lower interval end is -13.36 points (Qwen3-32B), still -8.85 at 1,000 honest runs, against a -10 bar; Seed-OSS-36B full width 15.39 points now, 8.88 at 1,000. So the count-matched 1% comparisons were too wide to support a 'matches' claim.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_robust_1pct_POSTHOC](../2026-10-13_robust_1pct_POSTHOC/) (method and module src/probe_shift/tail_threshold.py); [2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (count-matched comparisons)

## Main scripts

- [`run_cmcov_modal.py`](run_cmcov_modal.py)
- [`run_cmcov_splits_modal.py`](run_cmcov_splits_modal.py)
- [`run_poolsize_modal.py`](run_poolsize_modal.py)
- [`check_masks_modal.py`](check_masks_modal.py)

Result files the index names:

- `manifest_cmcov_results_v1.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_cmcov_splits_v1.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_poolsize_v1.json` (not copied: result left out to keep the release under 120 MB)
- [`mask_check_2026-10-13_183839.json`](results/mask_check_2026-10-13_183839.json)

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_countmatched_coverage_POSTHOC/` at commit `ee04c0ea5060c1baaa2e0c4a370459284ff5ef7d` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
