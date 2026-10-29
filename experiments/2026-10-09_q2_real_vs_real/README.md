# 2026-10-09_q2_real_vs_real

2026-10-09 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Real-versus-real controls: how much do two independent real runs per task differ in honest main-task success and in the impactful probe's catch and false alarms?

## What we did

Compared Qwen3.5-27B's two real honest runs per task (honest_a vs honest_b, strict rerun): main-task success on figure 1's 18 tasks, and the impactful-command probe's catch and false alarms on 20 tasks.

## What we found

One run passed 12 of 18 tasks and the other 11 of 18 (+5.6 points [0.0, +16.7]). The two honest batches catch 75.98% vs 70.79% (+5.19 points [-0.39, +9.49]) and false-alarm 4.77% vs 9.53% (descriptive).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-08_q2_figure_pack](../2026-10-08_q2_figure_pack/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`analyze_fig1_real_vs_real.py`](analyze_fig1_real_vs_real.py)
- [`run_onpolicy_real_vs_real.py`](run_onpolicy_real_vs_real.py)
- [`analyze_onpolicy_real_vs_real.py`](analyze_onpolicy_real_vs_real.py)

Result files the index names:

- `onpolicy_real_vs_real_analysis_2026-10-10_110300.json` (not in this folder at the source commit)
- `manifest_v2.json` (not copied: strings over 300 characters)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_q2_real_vs_real/` at commit `f09caded6889f3a06b710f113e25509b900b320d` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
