# 2026-10-13_q2_mixed_textalone

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

The mixed-recipe probe (trained on simulated attacks vs real honest runs) still overstates catch on simulated runs after every formatting repair: does that leftover cue survive when each tool output is re-read with only its own call and no history?

## What we did

Scored the frozen Seed-OSS-36B mixed-recipe probes from the simulated-control-evaluation probe study (one trained on unrepaired simulated runs, one after all formatting repairs) on stored activations read two ways, with full history (the new-reader test's activations) and as output plus its own call with no history (the causal test's no-cache activations), over 3,807 runs (real honest, first and second real attack, simulated attack), with a later label-shuffle null grouped by main task. Modal, small.

## What we found

mixed_textalone_POSTHOC_2026-10-13_173607.txt: the mixed probe trained on unrepaired simulated attacks versus real honest runs, run-level AUROC share kept without history 0.804 [0.774, 0.833], reading 'text (firm)'; in history simulated vs second real AUROC 0.968 and catch gap +76.0 points at 1%; without history 0.876 and +57.5. NOTE_labels: read 'text' as 'output plus its own call, no history'; the shuffle null moves shares by at most +0.003.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_g3_causal](../2026-10-13_g3_causal/) (no-history dump); [2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/) (dumps); [2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (another working session's mixed probes)

## Main scripts

- [`score_mixed.py`](score_mixed.py)
- [`score_mixed_fx.py`](score_mixed_fx.py)
- [`analyze_mixed.py`](analyze_mixed.py)
- [`recompute_fx_headline.py`](recompute_fx_headline.py)
- [`run_shuffle_null.py`](run_shuffle_null.py)

Result files the index names:

- `mixed_textalone_POSTHOC_2026-10-13_173607.txt` (not copied: lines over 300 characters)
- `mixed_textalone_fx_POSTHOC_2026-10-13_182320.txt` (not copied: lines over 300 characters)
- [`recompute_fx_headline_2026-10-13_183205.json`](results/recompute_fx_headline_2026-10-13_183205.json)
- [`shuffle_null_POSTHOC_2026-10-14_111544.json`](results/shuffle_null_POSTHOC_2026-10-14_111544.json)

## Files here

5 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_q2_mixed_textalone/` at commit `ff6d9304cd5f6bffd93bc54ee9d08bf99988762a` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
