# 2026-10-10_q2_false_claims

2026-10-10 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does the frozen deception probe score Qwen3.5-27B's false success claims (final reports saying done when the scorer says failed) higher than truthful ones, and alike in simulated runs?

## What we did

Plan and analysis committed before scoring; scored final reports of honest runs with the frozen step-2 deception probe, with a length check, task-block permutation, planted check and post hoc by-arm analysis.

## What we found

README: inconclusive pilot; AUROC 0.71 [0.46, 0.96] on 7 false vs 22 truthful claims (p = 0.11), length alone 0.73; real vs simulated claims 0.54.

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-09_final_report_claims](../2026-10-09_final_report_claims/); [2026-10-09_q2_deception_apollo](../2026-10-09_q2_deception_apollo/)

## Main scripts

- [`build_index.py`](build_index.py)
- [`run_score_claims.py`](run_score_claims.py)
- [`analyze_false_claims.py`](analyze_false_claims.py)
- [`analyze_length_check.py`](analyze_length_check.py)
- [`analyze_POSTHOC_by_arm.py`](analyze_POSTHOC_by_arm.py)

Result files the index names:

- [`false_claims_analysis_2026-10-10_160736_v2.json`](results/false_claims_analysis_2026-10-10_160736_v2.json)
- [`false_claims_length_check_2026-10-10_160736.json`](results/false_claims_length_check_2026-10-10_160736.json)
- [`false_claims_by_arm_POSTHOC_2026-10-11_194335.json`](results/false_claims_by_arm_POSTHOC_2026-10-11_194335.json)

## Files here

6 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-10_q2_false_claims/` at commit `3e9ea2fa1cf0003619a15baa4f538a8d2cd36569` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
