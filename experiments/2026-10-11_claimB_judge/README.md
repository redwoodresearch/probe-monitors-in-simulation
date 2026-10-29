# 2026-10-11_claimB_judge

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Can a zero-shot AI judge (GPT-5.6-Sol), reading one run at a time, tell formatting-fixed simulated runs from real ones as well as the main realism probe of the fixed-simulator test?

## What we did

Priced and piloted, then ran GPT-5.6-Sol through the batch API on 200 formatting-fixed simulated and 200 real held-out test runs plus planted runs and a retest, asking for the probability that each run is simulated, and compared its ranking score (AUROC) with the probe's on the same runs.

## What we found

Commit 8e5637624: judge 0.558 vs probe 0.821 (difference -0.263 [-0.320, -0.207]), so the judge is at least 0.10 below the probe; two real runs 0.492; 0 refusals; \$59.56.

## In the post?

Not in the post.

## Status

Exploratory (follow-up in [2026-10-11_claimB_judge_sweep](../2026-10-11_claimB_judge_sweep/))

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (its first probe step's runs and scores)

## Main scripts

- [`run_judge_pilot.py`](run_judge_pilot.py)
- [`run_judge_batch.py`](run_judge_batch.py)
- [`analyze_judge_run.py`](analyze_judge_run.py)
- [`analyze_judge_power.py`](analyze_judge_power.py)
- [`analyze_judge_price.py`](analyze_judge_price.py)
- [`analyze_judge_sizing.py`](analyze_judge_sizing.py)

Result files the index names:

- `judge_run_2026-10-11_150859.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `judge_run_2026-10-11_150859.png` (not copied: result file type not copied (.png))
- `judge_pilot_2026-10-11_124757.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `judge_power_2026-10-11_115302.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `judge_price_2026-10-11_115441.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

11 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_judge/` at commit `0bb9b474a6483d1e2a4ce2ec7b96c52f511a02df` of our internal repository. 17 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
