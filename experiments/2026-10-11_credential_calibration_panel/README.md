# 2026-10-11_credential_calibration_panel

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

The credential-monitor study: how does the five-model labelling panel's rate of outputs containing passwords or secrets compare with the original regex screening labels on the same runs, so the new-run gate can be calibrated?

## What we did

Ran the same five-judge LLM panel (Gemini 3.1 Pro, GPT-5.4, DeepSeek V4 Pro, Kimi K3, Grok 4.7 through OpenRouter; three agreeing votes) on 1,788 tool outputs from 59 original real honest GPT-5.6-Sol runs, with blind adjudication, and computed the calibration ratio for the committed gate.

## What we found

README and calibration_arithmetic.json: panel 131 positives (2.22 per run) vs regex 177 (3.0 per run), ratio 0.740; calibrated floor 2.1138, so stage one's 2.11 per run narrowly fails; about \$23.3.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_credential_stage1_labels](../2026-10-11_credential_stage1_labels/); [2026-10-11_q2_new_real_runs](../2026-10-11_q2_new_real_runs/)

## Main scripts

- [`run_judges.py`](run_judges.py)
- [`analyze_calibration.py`](analyze_calibration.py)

Result files the index names:

- [`summary.json`](results/final_labels/summary.json)
- `calibration_arithmetic.json` (not copied: result left out to keep the release under 120 MB)
- `accounting_verified.json` (not copied: result left out to keep the release under 120 MB)
- [`adjudication_audit.json`](results/final_labels/adjudication_audit.json)
- `screening_labels_calibration_panel.csv` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

10 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_credential_calibration_panel/` at commit `202ecf6f89c2dd5497568b8613f586cd9b58ac64` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
