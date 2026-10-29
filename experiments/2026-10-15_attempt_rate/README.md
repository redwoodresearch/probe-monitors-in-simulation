# 2026-10-15_attempt_rate

2026-10-15 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Does the agent clearly attempt the side task as often in simulation as on real machines, and are runs without an attempt longer?

## What we did

Counted attempt labels (attempted / unclear / declined) per attack run for simulated, real run 1 and real run 2 on all launched tasks (four task sets, 1,360 launched cells), with main-task bootstrap intervals and McNemar tests; plus run length (tool calls) for attempted vs not attempted runs. Keys and labels only, Modal CPU.

## What we found

Pooled launched cells: clear attempt 95.4% sim, 96.6% run 1, 96.1% run 2; sim minus run 1 -1.3 points (95% -2.6 to +0.1) (attempt_rate4 printout). Runs without an attempt ran near the step cap (sim median 192 tool calls on the pilot set, run_length printout).

## In the post?

In the post: Appendix G and footnote on attempts (via [2026-10-16_behaviour_full](../2026-10-16_behaviour_full/), which gates on this record)

## Status

Current (definition reused by behaviour_full)

## Builds on

[2026-10-14_q3_overnight_chain](../2026-10-14_q3_overnight_chain/) attempt labels; [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) membership710

## Main scripts

- [`run_attempt_rate.py`](run_attempt_rate.py)
- [`run_attempt_rate4.py`](run_attempt_rate4.py)
- [`run_run_length.py`](run_run_length.py)

Result files the index names:

- `attempt_rate_2026-10-15_234954.json` (not copied: strings over 300 characters)
- `attempt_rate4_2026-10-16_030654.json` (not copied: strings over 300 characters)
- `run_length_2026-10-16_000059.json` (not copied: result left out to keep the release under 120 MB)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 3 result files the index names, they match in 6 script-and-commit pairs, and in 3 the script was not in the folder at that commit.

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_attempt_rate/` at commit `0659538f16f953973d1a717bf9a48932c66cbce0` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
