# 2026-10-08_q3_verbalized_awareness

2026-10-08 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Does Qwen3.5-27B say in its own reasoning that it suspects simulation, evaluation, monitoring or real stakes, and does that differ between real and simulated runs? (pre-registered secondary)

## What we did

Agent-coded, arm-masked reading of all 14 Phase 2 runs that existed (5 real attack, 5 simulated attack, honest runs) with a rubric frozen before coding; descriptive only, no tests.

## What we found

Simulation came up in 1 of 5 simulated attacks (hedged) and 0 of 5 real; monitoring in 4 of 5 real and 5 of 5 simulated attacks (all instructed concealment); coders guessed the world for 8 of 10 attack runs.

## In the post?

Not in the post (the post's footnote on rarely verbalized evaluation awareness comes from the later [2026-10-14_q2_evalaware_step1](../2026-10-14_q2_evalaware_step1/)).

## Status

Superseded by [2026-10-14_q2_evalaware_step1](../2026-10-14_q2_evalaware_step1/) (verbalized awareness at scale, the source of the post's footnote)

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (runs); [2026-10-07_phase2_probe_proposal](../2026-10-07_phase2_probe_proposal/) (rule 2)

## Main scripts

- [`prepare_coding.py`](prepare_coding.py)
- [`agreement.py`](agreement.py)
- [`analyze_unmask.py`](analyze_unmask.py)
- [`scrub_for_git.py`](scrub_for_git.py)

Result files the index names:

- `verbalized_awareness_summary_2026-10-08_022150.json` (not copied: result left out to keep the release under 120 MB)
- `RUBRIC.md` (not copied: notes or README (internal working labels))

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_q3_verbalized_awareness/` at commit `0a42c96efe76723e66c4bd77a8e5da24c065f4e4` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
