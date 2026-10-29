# 2026-10-15_cmd_sim_detector

2026-10-15 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Can a linear detector on Qwen3.5-27B's activations at the end of each command (not only at tool outputs) tell simulated runs from real ones?

## What we did

On the 196-task pilot cohort (first 32,768 tokens only), fit L2 logistic-regression detectors at layers 48 and 63 for simulated vs real attack runs, real run 2 vs run 1 (control) and simulated vs real honest runs, with task-bootstrap intervals and a held-out run 2; H100 fits on Modal (about \$1.7).

## What we found

Run-level AUROC, simulated vs real run 1 attack: tool outputs (reply) layer 48 0.796 vs 0.495 for run 2 vs run 1; commands layer 48 0.593 vs 0.545 (attack_printout_1232.txt).

## In the post?

Not in the post.

## Status

Superseded by [2026-10-16_cmd_sim_detector_uncut](../2026-10-16_cmd_sim_detector_uncut/) and then [2026-10-16_cmd_sim_detector_uncut_all](../2026-10-16_cmd_sim_detector_uncut_all/) / [2026-10-16_attack_refit3](../2026-10-16_attack_refit3/) (all tasks, full length)

## Builds on

[2026-10-14_cmd_readout_simh_control](../2026-10-14_cmd_readout_simh_control/) (outside range)

## Main scripts

- [`run_attack_heldout.py`](run_attack_heldout.py)
- [`run_honest_v2.py`](run_honest_v2.py)
- [`plot_detector.py`](plot_detector.py)

Result files the index names:

- `attack_heldout_2026-10-15_122057.json` (not copied: result left out to keep the release under 120 MB)
- `honest_detector_2026-10-15_122054.json` (not copied: result left out to keep the release under 120 MB)
- `detector_auroc_2026-10-15_122057.png` (not copied: result file type not copied (.png))

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_cmd_sim_detector/` at commit `648f33e6115ebbe19276981dcbd79f9274d9f1ba` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
