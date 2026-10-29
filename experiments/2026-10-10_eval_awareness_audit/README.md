# 2026-10-10_eval_awareness_audit

2026-10-10 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Do Stage 1's evaluation-awareness training pairs contain word or length shortcuts?

## What we did

Step-0 audit of the Stage 1 contrast pairs: word-only and length-only classifiers per family, predictions written first.

## What we found

Yes, word shortcuts; in one family 'simulated' appears on the evaluation side in 123 of 132 pairs vs 0 of 132 (word alone 0.966); length was not a shortcut (0.402).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-07_stage1_awareness](../2026-10-07_stage1_awareness/)

## Main scripts

- [`analyze_stage1_pairs.py`](analyze_stage1_pairs.py)

Result files the index names:

- [`stage1_pairs_audit_2026-10-10_003704.json`](results/stage1_pairs_audit_2026-10-10_003704.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-10_eval_awareness_audit/` at commit `792d9905b99789411d7a957f99d55caaa7edc049` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
