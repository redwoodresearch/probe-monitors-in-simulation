# 2026-10-14_q2_evalaware_step1

2026-10-14 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

How often does Qwen3.5-27B's own reasoning say it is being tested, evaluated or simulated, in real vs simulated runs of the same tasks, and are there enough cases to train a probe?

## What we did

Scripted lexical-marker counts (strict and broad patterns) over the acting model's reasoning in real runs 1 and 2, simulated attack runs and honest arms (the pilot cohort with corrected Qwen3.5-27B transcript rendering, then simulated honest), with an LLM-judge validation on honest arms only and a within-task permutation comparison.

## What we found

Too rare to compare: of 190 common tasks only 2 were informative; reading "too few informative tasks"; step-2 verdict "not enough positives"; strict markers in 2 of 191 runs of the first real honest arm (evalaware_first_2026-10-14_212550.json).

## In the post?

In the post: footnote on evaluation awareness ([^aware]: the agent rarely said it was being evaluated)

## Status

Current

## Builds on

[2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/) (corrected-B manifest); [2026-10-14_q3_q35_simhonest_purchase](../2026-10-14_q3_q35_simhonest_purchase/)

## Main scripts

- [`run_evalaware.py`](run_evalaware.py)
- [`launch_deployed.py`](launch_deployed.py)
- [`power_paired.py`](power_paired.py)
- [`spotread_yes.py`](spotread_yes.py)
- [`schema_probe.py`](schema_probe.py)
- [`evalzip.py`](evalzip.py)

Result files the index names:

- `evalaware_first_2026-10-14_212550.json` (not copied: result left out to keep the release under 120 MB)
- `evalaware_simh_final_2026-10-14_230446.json` (not copied: result left out to keep the release under 120 MB)
- `power_paired_2026-10-14_230447.json` (not copied: result left out to keep the release under 120 MB)
- [`spotread_yes_2026-10-14_233226.json`](results/spotread_yes_2026-10-14_233226.json)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/4f58f5640/run_evalaware.py`](as_run/4f58f5640/run_evalaware.py) from commit `4f58f5640`: results/evalaware_first_2026-10-14_212550.json, the source of the post's footnote on evaluation awareness, records commit 4f58f5640; at that commit this script differs from the copy in this folder's snapshot, which is newer.
- [`as_run/4f58f5640/launch_deployed.py`](as_run/4f58f5640/launch_deployed.py) from commit `4f58f5640`: results/evalaware_first_2026-10-14_212550.json, the source of the post's footnote on evaluation awareness, records commit 4f58f5640; at that commit this script differs from the copy in this folder's snapshot, which is newer.
- [`as_run/746426379/spotread_yes.py`](as_run/746426379/spotread_yes.py) from commit `746426379`: results/evalaware_simh_final_2026-10-14_230446.json records commit 746426379; at that commit this script differs from the copy in this folder's snapshot, which is newer.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 2 result files the index names, they match in 7 script-and-commit pairs, differ in 3, and in 2 the script was not in the folder at that commit. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

6 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_q2_evalaware_step1/` at commit `d997f402672b919f3ddd59925ec2a2dcad8d5436` of our internal repository. 17 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 3 files under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
