# 2026-10-20_routec_rerun_refit3-l48

2026-10-20 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Check the layer-48 simulation-detector values (Figure 2, Table 7, and the off-policy drop-vs-detector correlation) by refitting and replaying.

## What we did

Value-half check: certificates, an independent refit and a stored-weights replay of the command layer-48 detector cell (tool outputs as diagnostic), plus a rank-identity test for the Spearman rows.

## What we found

Command layer 48: REPRODUCED, pass true (NOTE_refit3l48_check).

## In the post?

Not in the post (verifies Figure 2, Table 7 and Appendix C's Spearman correlation).

## Status

Current

## Builds on

[2026-10-16_attack_refit3](../2026-10-16_attack_refit3/)

## Main scripts

- [`run_refit3_l48_check.py`](run_refit3_l48_check.py)
- [`score_predictions.jq`](results/score_predictions.jq)

Result files the index names:

- `refit3l48_check_2026-10-20_142501_ap-TO5rONSc5UrYGItBZtw3cw.json` (not copied: result file 1 MB or larger)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_rerun_refit3-l48/` at commit `9e16a371bbaf86d6073c462ed945f94f5a17901c` of our internal repository. 16 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
