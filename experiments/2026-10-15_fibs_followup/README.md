# 2026-10-15_fibs_followup

2026-10-15 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does the deception probe work better when read at the model's answer to a follow-up question, as its paper does, than at commands or tool outputs?

## What we did

Pilot on 20 real attack + 20 real honest runs: appended follow-up questions with prefilled answers, scored the deception probe there; then a specificity check and a neutral-question control.

## What we found

Follow-up readout AUROC 0.973 [0.910, 1.000] vs 0.760 at commands; neutral-question controls 0.527 and 0.340, so the signal is question-specific (analysis printouts). Caveats: the specificity check reads "in between" (folded median 0.693), and RULE.md notes the follow-up readout is structurally the end-of-sequence readout the project rules ban.

## In the post?

Not in the post.

## Status

Exploratory (readout not used in the post)

## Builds on

2026-10-14 deception-probe score record

## Main scripts

- [`run_fibs_followup.py`](run_fibs_followup.py)
- [`run_neutral_control.py`](run_neutral_control.py)
- [`run_specificity.py`](run_specificity.py)
- [`analyze_fibs_followup.py`](analyze_fibs_followup.py)

Result files the index names:

- `fibs_followup_analysis_2026-10-15_121932.json` (not copied: result left out to keep the release under 120 MB)
- `fibs_neutral_analysis_2026-10-16_002650.json` (not copied: result left out to keep the release under 120 MB)

## Files here

5 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-15_fibs_followup/` at commit `c236b5d5b256e405a1c7114c50e0e3c83b601f80` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
