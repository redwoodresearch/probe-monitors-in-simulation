# 2026-10-12_destructive_transfer_labels

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Which of Qwen3.5-27B's acting commands delete or overwrite content, or send data off the machine (labels for destructive-command and data-transfer monitors)?

## What we did

Labelled 74,873 command calls from six arms of the 196-task acting runs two ways: exact pattern rules (the five-column handoff) and a blinded panel of AI judges (Gemini 3.1 Pro, GPT-5.4, DeepSeek V4 Pro, Kimi K3, Grok 4.7, plus high-reasoning reviewers) under a written decision rule, with synthetic controls and repairs.

## What we found

README and RUN.md: the pattern-rule handoff is complete (2,275 destructive and 2,628 transfer positives; 27 planted transfer checks pass); the AI-judge labels are provisional (72,922 complete, 1,951 unresolved); reported charges \$821.53 (\$929.89 conservative).

## In the post?

Not in the post.

## Status

Exploratory (pattern labels final; AI-judge labels provisional, not ready for fitting)

## Builds on

[2026-10-12_dt_acting_monitors](../2026-10-12_dt_acting_monitors/) (label sheet)

## Main scripts

- [`run_judges.py`](run_judges.py)
- [`analyze_judges.py`](analyze_judges.py)
- [`analyze_rules.py`](analyze_rules.py)
- [`analyze_rule_handoff.py`](analyze_rule_handoff.py)
- [`analyze_finish.py`](analyze_finish.py)
- [`analyze_verify.py`](analyze_verify.py)

Result files the index names:

- `label_decision_rule_v2.md` (not copied: notes or README (internal working labels))
- `RUN.md` (not copied: notes or README (internal working labels))

## Files here

44 code and configuration files and 26 result files, copied byte for byte from `experiments/2026-10-12_destructive_transfer_labels/` at commit `57ad8f71ae717ed9a81181c802395a44cbdc6033` of our internal repository. 81 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
