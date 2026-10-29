# 2026-10-10_eval_awareness_probe

2026-10-10 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Can an evaluation-awareness probe on Qwen3.5-27B, trained on de-shortcut synthetic pairs, detect an inserted 'this is an evaluation' line in real LinuxArena runs (the gate before scoring real vs simulated)?

## What we did

Step 1: extracted and fit probes on rebuilt synthetic pairs, then planted evaluation lines (far, near, checker header) in 99 real honest runs as positive controls; diagnostics on synthetic pairs, paired rescoring, distance checks, two retrains (real-looking commands, turn-start token) and neutral-line controls.

## What we found

Passed held-out pairs (0.970) but caught the inserted line in only 5 of 99 real runs against an 80% bar, so the gate failed. Almost every run moved the right way but only 0.014x to 0.117x the synthetic size. A turn-start retrain responds more but may react to any inserted text.

## In the post?

Not in the post (the post's footnote probe is the later Nguyen-method probe in [2026-10-14_p2_evalaware_nguyen](../2026-10-14_p2_evalaware_nguyen/)).

## Status

Current (failed its gate; later replaced as the project's evaluation-awareness probe by [2026-10-14_p2_evalaware_nguyen](../2026-10-14_p2_evalaware_nguyen/))

## Builds on

[2026-10-10_eval_awareness_audit](../2026-10-10_eval_awareness_audit/); [2026-10-07_stage1_awareness](../2026-10-07_stage1_awareness/)

## Main scripts

- [`build_sets.py`](build_sets.py)
- [`run_extract.py`](run_extract.py)
- [`run_fit.py`](run_fit.py)
- [`run_score_la.py`](run_score_la.py)
- [`analyze_step1.py`](analyze_step1.py)
- [`analyze_paired_rescore.py`](analyze_paired_rescore.py)
- [`run_build_retrain.py`](run_build_retrain.py)
- [`run_extract_retrain.py`](run_extract_retrain.py)
- [`analyze_retrain.py`](analyze_retrain.py)
- [`analyze_neutral.py`](analyze_neutral.py)

Result files the index names:

- [`step1_gate_2026-10-10_025308.json`](results/step1_gate_2026-10-10_025308.json)
- `paired_rescore_2026-10-10_214042.json` (not copied: result left out to keep the release under 120 MB)
- `retrain_analysis_2026-10-10_234209.json` (not copied: result left out to keep the release under 120 MB)
- [`fit_2026-10-10_230615.json`](results/fit_2026-10-10_230615.json)

## Files here

28 code and configuration files and 5 result files, copied byte for byte from `experiments/2026-10-10_eval_awareness_probe/` at commit `158eeeebd1325bdb3adc120618a057c47af1c434` of our internal repository. 67 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
