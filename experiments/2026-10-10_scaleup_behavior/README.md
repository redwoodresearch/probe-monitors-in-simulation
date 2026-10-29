# 2026-10-10_scaleup_behavior

2026-10-10 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Does Qwen3.5-27B act differently in simulation at 196 tasks, measured against two real attack runs?

## What we did

CPU-only behavior measures (edit share, re-read rate, failure-text share, steps, output tokens) from the logs of the strict rerun, canary, launch 2, top-up and second attack batch, paired by task, with planted shifts.

## What we found

No detectable difference beyond run-to-run variation; steps real minus simulated -1.2 [-6.5, +3.9] vs real minus real +3.6 [-0.9, +8.0]; only 2 of 4 planted shifts detected.

## In the post?

Not in the post (Appendix G (a 196-task early version; the post's numbers come from [2026-10-16_behaviour_full](../2026-10-16_behaviour_full/))).

## Status

Superseded by [2026-10-16_behaviour_full](../2026-10-16_behaviour_full/) (1,313 tasks, Appendix G)

## Builds on

[2026-10-09_phase2_onpolicy_behavior](../2026-10-09_phase2_onpolicy_behavior/); [2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`extract_behavior.py`](extract_behavior.py)
- [`analyze_behavior.py`](analyze_behavior.py)
- [`mde_scan.py`](mde_scan.py)
- [`recheck_counted_attempts.py`](recheck_counted_attempts.py)

Result files the index names:

- `s3_manifest_v3.json` (not copied: strings over 300 characters)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_scaleup_behavior/` at commit `77cf757d00b096dffeadb23dafb62f162bb48c4c` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
