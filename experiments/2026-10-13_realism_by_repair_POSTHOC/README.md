# 2026-10-13_realism_by_repair_POSTHOC

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does patching the simulator's reply text remove the simulated-vs-real signal in honest runs?

## What we did

Nested-fold realism probes (Qwen3-32B tool-output readout, layers 48 and 63) separating simulated from real honest runs at each repair level (as simulated, the editor-output fixes, all formatting fixes, newline fixes only, and all fixes plus the file-view newline fix) against a real-vs-real honest control on 431 tasks; plus a GPU label-shuffle null and follow-ups. Modal CPU and GPU, \$0.74 plus nulls.

## What we found

realism_by_repair_POSTHOC_2026-10-13_174155.printout.txt: realism AUROC at layer 48 falls from 0.942 (as is) to 0.815 (the editor-output formatting repairs) and 0.808 (all formatting repairs), against a real-vs-real control of 0.493; at layer 63 0.971 to 0.889 and 0.858 (control 0.541); newline-only barely moves it. Patching lowers but does not remove the signal.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/) (repair arms); [2026-10-13_world_direction_POSTHOC](../2026-10-13_world_direction_POSTHOC/) (gate)

## Main scripts

- [`run_realism_by_repair.py`](run_realism_by_repair.py)
- [`run_realism_null.py`](run_realism_null.py)
- [`run_null_gpu.py`](run_null_gpu.py)
- [`run_fx_realism.py`](run_fx_realism.py)
- [`run_fx_followups.py`](run_fx_followups.py)
- [`plot_realism_ladder.py`](plot_realism_ladder.py)

Result files the index names:

- [`realism_by_repair_POSTHOC_2026-10-13_174155.printout.txt`](results/realism_by_repair_POSTHOC_2026-10-13_174155.printout.txt)
- [`null_gpu_full_20261013_222056.printout.txt`](results/null_gpu_full_20261013_222056.printout.txt)
- [`fx_20261013_224607.printout.txt`](results/fx_20261013_224607.printout.txt)
- [`fxfollow_20261013_230107.printout.txt`](results/fxfollow_20261013_230107.printout.txt)

## Files here

15 code and configuration files and 6 result files, copied byte for byte from `experiments/2026-10-13_realism_by_repair_POSTHOC/` at commit `a28ef31151b2c85c68920eecccd149ce68a784e8` of our internal repository. 24 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
