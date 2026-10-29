# 2026-10-08_q2_free_checks_3

2026-10-08 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

How often does GPT-5.6-Sol pass its real honest main tasks on the on-policy pilot's 20 tasks (the reference for comparing Qwen3.5-27B's competence)?

## What we did

Read main_task_success from each real honest GPT-5.6-Sol run's log on the 20 pilot tasks; the Sol side of figure 1 in the figure pack.

## What we found

README: GPT-5.6-Sol succeeded in 23 of 39 scored runs, 0.5897 (95% interval [0.421, 0.7443]).

## In the post?

Not in the post.

## Status

Current (input to [2026-10-08_q2_figure_pack](../2026-10-08_q2_figure_pack/))

## Builds on

None stated

## Main scripts

- [`analyze_honest_success.py`](analyze_honest_success.py)

Result files the index names:

- [`honest_success_2026-10-08_173215.compact.json`](results/honest_success_2026-10-08_173215.compact.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-08_q2_free_checks_3/` at commit `f4c2506d0f46054b7c4176075c7eadcdeb6d369f` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
