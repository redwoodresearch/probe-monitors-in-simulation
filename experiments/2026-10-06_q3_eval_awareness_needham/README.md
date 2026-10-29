# 2026-10-06_q3_eval_awareness_needham

2026-10-06 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Does an evaluation-awareness probe trained on real evaluation and deployment transcripts (Needham et al. 2025) score simulated LinuxArena runs differently from real ones?

## What we did

Phase 1: Qwen3-32B probes (all 64 layers, mean over assistant tokens) trained on 976 real evaluation vs deployment transcripts, checked on held-out sources and a planted 'you are being evaluated' line, then read on real and simulated GPT-5.6-Sol LinuxArena runs.

## What we found

No shift (simulated minus real +0.001 [-0.138, +0.141] vs real-to-real +0.105); README: the probe recognizes the type of source rather than evaluation cues (within one scaffold AUROC 0.29-0.41), so the null says little.

## In the post?

Not in the post.

## Status

Current (kept as a record; README calls it a weak LinuxArena monitor)

## Builds on

Not stated

## Main scripts

- [`run_fetch.py`](run_fetch.py)
- [`run_extract.py`](run_extract.py)
- [`run_fit.py`](run_fit.py)
- [`run_control.py`](run_control.py)
- [`run_score_la.py`](run_score_la.py)
- [`analyze_la.py`](analyze_la.py)
- [`analyze_control.py`](analyze_control.py)

Result files the index names:

- `fit_2026-10-06_180227.json` (not copied: result file 1 MB or larger)
- [`family_holdout_2026-10-06_181049.json`](results/family_holdout_2026-10-06_181049.json)
- [`control_2026-10-06_182738.json`](results/control_2026-10-06_182738.json)
- [`la_shift_2026-10-06_182724.json`](results/la_shift_2026-10-06_182724.json)

## Files here

9 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-06_q3_eval_awareness_needham/` at commit `b302e816b45748673a37fd6e3725711caea2f3a1` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
