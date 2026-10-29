# 2026-10-11_credential_stage1_labels

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

The credential-monitor study, stage 1: how many tool outputs in the 100 new real GPT-5.6-Sol honest runs (50 tasks, two runs each) visibly contain passwords or secrets?

## What we did

Labelled all 2,796 reader-visible outputs with a five-model LLM panel through OpenRouter (three confident agreeing votes, blind adjudication of the rest), compared with the unchanged regex screen, and ran an independent 50-row audit with a local disagreement viewer.

## What we found

README: 211 positives, 2.11 per run (regex screen 264, 2.64 per run); AUDIT.md: 50 of 50 sampled rows agree with an independent model audit (Wilson 92.9-100%).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_q2_new_real_runs](../2026-10-11_q2_new_real_runs/) (stage 1 runs)

## Main scripts

- [`run_judges.py`](run_judges.py)
- [`analyze_prepare.py`](analyze_prepare.py)
- [`analyze_labels.py`](analyze_labels.py)
- [`analyze_adjudication.py`](analyze_adjudication.py)
- [`analyze_screening_baseline.py`](analyze_screening_baseline.py)
- [`analyze_audit.py`](analyze_audit.py)

Result files the index names:

- `screening_labels_oct11c_stage1.csv` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

14 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-11_credential_stage1_labels/` at commit `d65b2591e736e067b97d8ee833cf246bb8902845` of our internal repository. 21 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
