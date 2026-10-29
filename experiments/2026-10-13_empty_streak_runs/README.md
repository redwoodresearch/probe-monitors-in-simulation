# 2026-10-13_empty_streak_runs

2026-10-13 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Why did 4 simulated attack runs break (long streaks of empty tool replies), and do such streaks also happen unflagged in real runs?

## What we did

Metadata-only script pass over the .eval files of Qwen3.5-27B's 196-task runs (first and second real attack, simulated attack, two real honest batches, 20 simulated honest), counting empty-reply streaks and the simulator's completions behind them; follow-ups on provenance and tool-span pairing. \$0, local.

## What we found

empty_streak_runs_POSTHOC_2026-10-13_103135.printout.txt: runs with a streak of 20 or more empty replies: simulated attack 4 of 191, first real attack 0 of 194, second real attack 1 of 194, real honest 1 and 3 of 195; README: every empty reply the simulator answered was the literal '&lt;OUTPUT>\n&lt;/OUTPUT>'.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

A side branch @ 9c5bcbca; the width_196 sensitivity ([2026-10-13_sce_width_POSTHOC](../2026-10-13_sce_width_POSTHOC/))

## Main scripts

- [`analyze_empty_streak_runs_POSTHOC.py`](analyze_empty_streak_runs_POSTHOC.py)
- [`analyze_sim_empty_completion_POSTHOC.py`](analyze_sim_empty_completion_POSTHOC.py)
- [`analyze_streak_followups_POSTHOC.py`](analyze_streak_followups_POSTHOC.py)
- [`analyze_span_window_check_POSTHOC.py`](analyze_span_window_check_POSTHOC.py)

Result files the index names:

- `empty_streak_runs_POSTHOC_2026-10-13_103135.printout.txt` (not copied: lines over 300 characters)
- [`print_sim_empty_completion_POSTHOC_at_46fc5315.printout.txt`](results/print_sim_empty_completion_POSTHOC_at_46fc5315.printout.txt)
- [`streak_followups_POSTHOC_2026-10-13_110755.printout.txt`](results/streak_followups_POSTHOC_2026-10-13_110755.printout.txt)
- `span_window_check_POSTHOC_at_fe755ab8.printout.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

4 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_empty_streak_runs/` at commit `a37d414c0677b9fb55d5fb22d1f2a585942f33df` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
