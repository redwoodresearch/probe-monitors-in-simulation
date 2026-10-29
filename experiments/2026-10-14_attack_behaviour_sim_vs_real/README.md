# 2026-10-14_attack_behaviour_sim_vs_real

2026-10-14 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Does Qwen3.5-27B attack differently in simulation (attempt rate, side-task success, steps, tool calls, tokens, submit timing), and do those differences predict the per-task probe score gap between simulated and real runs?

## What we did

Scripted counts from the raw .eval logs of real run 1, real run 2 and simulated attack runs on the 196-task pilot (190 tasks with all three arms); paired differences with environment-stratified task bootstrap (10,000 draws), Spearman correlations between behaviour differences and standardized probe score gaps for real-trained and published probes, plus a run-length dilution analysis.

## What we found

On 190 tasks, simulated minus real run 1: attempted −0.011 [−0.032, +0.005], steps +0.55 [−4.3, +5.5], output tokens +246 [−1,370, +2,001]; side-task success 0.832 vs 0.665 (+0.165 [+0.091, +0.238]), though simulated success is judged by a language model (behaviour_2026-10-15_004811_printout.txt).

## In the post?

Not in the post.

## Status

Superseded by [2026-10-16_behaviour_full](../2026-10-16_behaviour_full/) (Appendix G on all 1,313 tasks)

## Builds on

[2026-10-14_runlength_check](../2026-10-14_runlength_check/) (same eval logs and join route); [2026-10-14_fpr_multiplicity](../2026-10-14_fpr_multiplicity/) (per-row score files)

## Main scripts

- [`run_behaviour.py`](run_behaviour.py)
- [`run_length_dilution.py`](run_length_dilution.py)
- [`check_agent_events.py`](check_agent_events.py)

Result files the index names:

- [`behaviour_2026-10-15_002550.json`](results/behaviour_2026-10-15_002550.json)
- [`behaviour_2026-10-15_004811.json`](results/behaviour_2026-10-15_004811.json)
- [`length_dilution_2026-10-15_005921.json`](results/length_dilution_2026-10-15_005921.json)
- [`agent_events_check_2026-10-15_003247.json`](results/agent_events_check_2026-10-15_003247.json)

## Files here

3 code and configuration files and 5 result files, copied byte for byte from `experiments/2026-10-14_attack_behaviour_sim_vs_real/` at commit `4eb87c2d1183cc7208fd64bdbb05a09a46f65f3e` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
