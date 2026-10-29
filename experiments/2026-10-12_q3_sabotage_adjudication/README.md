# 2026-10-12_q3_sabotage_adjudication

2026-10-12 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

How does the stage-1 sabotage probe's recall change when scored against blinded adjudicated labels instead of the AI judge panel's labels?

## What we did

Drew a stratified blinded sheet of 100 calls for adjudication, re-scored the stage-1 sabotage gate (Qwen3.5-27B, layer 30) on the adjudicated labels including calls the gate never scored, and saved a refit of the gate's weights before its activations expired (\$0.0018).

## What we found

Recall on adjudicated labels 12.4% [0.0, 33.0], -13.1 points vs panel recall [-25.5, +7.5]; AUROC 0.861 [0.710, 0.965]; the refit reproduces the stored gate exactly and its weights are now saved.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-11_sabotage_stage1_labels](../2026-10-11_sabotage_stage1_labels/)

## Main scripts

- [`make_adjudication_sheet.py`](make_adjudication_sheet.py)
- [`rescore_adjudication.py`](rescore_adjudication.py)
- [`run_score_unscored.py`](run_score_unscored.py)
- [`analyze_unscored.py`](analyze_unscored.py)
- [`freeze_llm_return.py`](freeze_llm_return.py)

Result files the index names:

- [`adjudication_sheet_2026-10-12_142404.csv`](results/adjudication_sheet_2026-10-12_142404.csv)

## Files here

7 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-12_q3_sabotage_adjudication/` at commit `6cac014e070c63e93f681833ffb41bd19fdf5e48` of our internal repository. 20 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
