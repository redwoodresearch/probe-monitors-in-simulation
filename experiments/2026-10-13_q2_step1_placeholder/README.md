# 2026-10-13_q2_step1_placeholder

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Step 1 of the test of why repaired probes still overstate catch (agent behaviour alone): with every tool reply replaced by one placeholder, can a probe still tell simulated from real runs, or attack from honest?

## What we did

Staged GPT-5.6-Sol runs (first and second real attack, real honest, simulated attack and honest) with every tool message replaced by '[tool output withheld]' and extracted Qwen3-32B layers 48 and 63 at the end of every tool call and tool output (4,244 transcripts), with staging and bit-exact reproduction checks. Fits were planned with the training recipes of the simulated-control-evaluation probe study; about \$36 to \$47 estimated.

## What we found

Only staging and extraction checks are committed: check_repro (17:49) says the one flagged run has no step-0 call in either extraction, and 765 sampled rows are bit-exact on two GPU types. The fits ran in [2026-10-13_why_steps](../2026-10-13_why_steps/): with placeholder replies, the fully repaired mixed probe's overstatement at the command readout (layer 48, 1%) is +2.2 points [-0.9, +5.2], against +5.6 [+2.6, +8.6] with the repaired replies (why_metrics_steps_2026-10-13_175358.json, rows L48__ph_call_end__c and L48__ff_call_end__c).

## In the post?

Not in the post.

## Status

Exploratory (data for Step 1; analysis in [2026-10-13_why_steps](../2026-10-13_why_steps/))

## Builds on

[2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/) (stager sources); the attack detector's extractor

## Main scripts

- [`stage_placeholder.py`](stage_placeholder.py)
- [`run_extract_qwen3_ph.py`](run_extract_qwen3_ph.py)
- [`check_ph.py`](check_ph.py)
- [`check_ph_repro.py`](check_ph_repro.py)

Result files the index names:

- [`stage_2026-10-13_171415.json`](results/stage_2026-10-13_171415.json)
- [`check_2026-10-13_step1ph_qwen3-32b_l48-63_v1_2026-10-13_174034.json`](results/check_2026-10-13_step1ph_qwen3-32b_l48-63_v1_2026-10-13_174034.json)
- [`check_repro_2026-10-13_step1ph_qwen3-32b_l48-63_v1_2026-10-13_174945.json`](results/check_repro_2026-10-13_step1ph_qwen3-32b_l48-63_v1_2026-10-13_174945.json)
- `s3_manifest_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

5 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-13_q2_step1_placeholder/` at commit `12fb19fbcdb34de9afe6e03209fce1dd2ac83228` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
