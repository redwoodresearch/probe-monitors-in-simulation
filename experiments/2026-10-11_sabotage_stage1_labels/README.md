# 2026-10-11_sabotage_stage1_labels

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Which of the 1,457 selected stage-one commands directly implement the hidden side task (labels for the sabotage-probe test)?

## What we did

Five-model LLM panel through OpenRouter (Gemini 3.1 Pro, GPT-5.4, DeepSeek V4 Pro, Kimi K3, Grok 4.7; four confident agreeing votes) judging each command with its full command sequence and blinded IDs, plus an independent audit and a local disagreement viewer.

## What we found

README: 86 sabotage, 1,283 not sabotage, 88 unclear; delivered to the return prefix with readback hash match; no downstream result claimed here. AUDIT.md: a blinded audit of 50 seeded labels agreed on 39 (78%; 39 of 41 where the panel did not abstain).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_sabotage_test](../2026-10-11_sabotage_test/) (stage-1 lists)

## Main scripts

- [`analyze_prepare.py`](analyze_prepare.py)
- [`run_judges.py`](run_judges.py)
- [`analyze_labels.py`](analyze_labels.py)
- [`analyze_audit.py`](analyze_audit.py)

Result files the index names:

- `labels_stage1_filled.csv` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- [`summary.json`](results/final_labels/summary.json)
- [`vote_audit.json`](results/final_labels/vote_audit.json)
- `judge_coverage.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

8 code and configuration files and 6 result files, copied byte for byte from `experiments/2026-10-11_sabotage_stage1_labels/` at commit `b90cd42a3f8669aa0f45fe952d454a5ef5389ed9` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
