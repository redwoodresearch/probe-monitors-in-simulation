# 2026-10-14_p2_evalaware_nguyen

2026-10-14 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Can an evaluation-awareness probe built with Nguyen et al.'s published method (arXiv 2507.01786) work on Qwen3.5-27B, i.e. respond when the agent is told it is being tested?

## What we did

Fit a mean-difference direction on Qwen3.5-27B activations from the method's data (layer 53 selected), then ran a planted-cue validation gate (runs with an explicit test statement) before any real-vs-simulated scoring; a fresh agent rechecked the record.

## What we found

Verdict "does not validate on Qwen3.5": stage 1 passed (layer 53 held-out AUROC 1.000), but the planted-cue gate failed, so no real-vs-simulated number was computed (evalaware_2026-10-14_162917.printout.txt; RULE.md result).

## In the post?

In the post: footnote on evaluation awareness ([^aware], main text "What we ran")

## Status

Current

## Builds on

[2026-10-10_eval_awareness_probe](../2026-10-10_eval_awareness_probe/) (earlier failed attempt)

## Main scripts

- [`run_evalaware.py`](run_evalaware.py)
- [`evalaware_lib.py`](evalaware_lib.py)
- [`recheck_evalaware.py`](recheck_evalaware.py)

Result files the index names:

- [`recheck_evalaware_2026-10-14_164703.json`](results/recheck/recheck_evalaware_2026-10-14_164703.json)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_p2_evalaware_nguyen/` at commit `9e305baf47b12e87d191cbffcc65cce45e222cfe` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
