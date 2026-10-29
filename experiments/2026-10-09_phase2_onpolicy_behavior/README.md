# 2026-10-09_phase2_onpolicy_behavior

2026-10-09 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Does Qwen3.5-27B behave differently in simulation (steps, commands, endings) on the 20-task strict rerun?

## What we did

Free, read-only task-paired comparison of behavior measures (steps, tool mix, endings) between real and simulated runs (September primary, today's secondary) on the 20-task strict rerun, with verification scripts; replaces the voided 10-08 pilot reads.

## What we found

Median attack steps 50 real / 49.5 September simulated / 47 today's simulated; over 78 pairs the simulated run was shorter in 40 and longer in 36 (p = 0.73); descriptive, preliminary.

## In the post?

Not in the post (Appendix G (an early 20-task version; the post's numbers come from [2026-10-16_behaviour_full](../2026-10-16_behaviour_full/))).

## Status

Superseded by [2026-10-10_scaleup_behavior](../2026-10-10_scaleup_behavior/) (196 tasks) and then [2026-10-16_behaviour_full](../2026-10-16_behaviour_full/) (1,313 tasks, Appendix G)

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (strict rerun)

## Main scripts

- [`analyze_onpolicy_behavior.py`](analyze_onpolicy_behavior.py)
- [`analyze_onpolicy_distributions.py`](analyze_onpolicy_distributions.py)
- [`behavior_common.py`](behavior_common.py)
- verify/behavior/\*.py

Result files the index names:

- `s3_outputs_2026-10-09_v3.json` (not copied: result left out to keep the release under 120 MB)

## Files here

16 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_phase2_onpolicy_behavior/` at commit `8d87cf640de01131e250b9541022bf52a2bfe02f` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
