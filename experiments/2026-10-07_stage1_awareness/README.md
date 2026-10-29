# 2026-10-07_stage1_awareness

2026-10-07 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Do evaluation-awareness, monitoring-awareness and high-stakes probes on Qwen3.5-27B pass validation on held-out synthetic framings before being scored on real and simulated LinuxArena runs?

## What we did

Pre-registered Stage 1: one H100 extraction at all 64 layers of synthetic contrast sets, per-layer logistic-regression probes on held-out framing families with label-swap nulls, read at the last command token; an impactful-command control probe at layer 48 was scored on Phase 1 real vs simulated runs.

## What we found

Every awareness probe failed validation (evaluation awareness held-out 0.733 against a 0.541 null; monitoring and stakes fell to 0.556 and 0.580 on implicit cues), so none was scored on LinuxArena. The control probe scored real and simulated alike (attack 0.503 [0.483, 0.522]).

## In the post?

Not in the post (the post's evaluation-awareness footnote comes from the later [2026-10-14_p2_evalaware_nguyen](../2026-10-14_p2_evalaware_nguyen/)).

## Status

Current (failed validation; direction dropped)

## Builds on

[2026-10-07_phase2_probe_proposal](../2026-10-07_phase2_probe_proposal/) (pre-registration)

## Main scripts

- [`build_sets.py`](build_sets.py)
- [`run_extract_stage1.py`](run_extract_stage1.py)
- [`run_fit_stage1.py`](run_fit_stage1.py)
- [`run_score_stage1.py`](run_score_stage1.py)
- [`analyze_fit_stage1.py`](analyze_fit_stage1.py)
- [`analyze_review_checks.py`](analyze_review_checks.py)
- [`run_drop_by_arm.py`](run_drop_by_arm.py)

Result files the index names:

- `freeze_2026-10-07_234608.json` (not copied: result left out to keep the release under 120 MB)
- `review_checks_2026-10-08_164308.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-07_stage1_awareness/` at commit `183e3849b5c3f1e28496af2c15e19f4743bc5fbf` of our internal repository. 16 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
