# 2026-10-13_crossfamily_label_audit

2026-10-13 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Do independent model families (not just GPT-5.6-Sol) agree with the meaning-judge labels (does a simulated reply mean the same as the real one?) and the sabotage labels on agent commands?

## What we did

Codex ran a blind audit: GPT-5.6-Sol, Gemini 3.1 Pro and DeepSeek (Claude partial, excluded) labeled 117 meaning items with a gate of known-breaking items and controls, and every command in 55 frozen sabotage packets (120 target items), with answers frozen before key scoring. API spend \$80.06.

## What we found

README: only Sol passed the meaning gate (31 of 33 breaking, 18 of 18 controls); Sol judged 22 of 33 simulator-real and 24 of 33 real-real pairs the same, difference -6.06 points [-15.63, 0.00], so the 10-point noninferiority claim is inconclusive; sabotage positives confirmed 19/19/18 of 20 (Sol), 20/20/20 (Gemini), 12/16/18 (DeepSeek) for simulated/real 1/real 2. No independent-family corroboration of Sol.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_claimB_meaning_judge](../2026-10-12_claimB_meaning_judge/) (meaning sheet); stage-1 sabotage labels ([2026-10-12_q3_sabotage_adjudication](../2026-10-12_q3_sabotage_adjudication/))

## Main scripts

- [`run_audit.py`](run_audit.py)
- [`score_frozen_models.py`](score_frozen_models.py)
- [`score_sabotage_committed.py`](score_sabotage_committed.py)

Result files the index names:

- `frozen_answers_2026-10-13_v2.json` (not copied: result left out to keep the release under 120 MB)

## Files here

12 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_crossfamily_label_audit/` at commit `c6f72ccd86eedcd33f0e0ee3e5c10cdb75e37aca` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
