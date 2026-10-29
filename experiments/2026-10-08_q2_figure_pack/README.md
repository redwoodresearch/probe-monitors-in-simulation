# 2026-10-08_q2_figure_pack

2026-10-08 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Figures for the project manager: how often do the old (Qwen3-32B) and new (Qwen3.5-27B) agents pass honest LinuxArena tasks with GPT-5.6-Sol as a reference, plus forest and refit figures for the impactful-command 2x2.

## What we did

Plots drawn from cited result files (nothing new computed), including the corrected figure 1 on 18 tasks with task-weighted success after removing a task with a placeholder grader.

## What we found

Task-weighted honest success Qwen3-32B 0%, Qwen3.5-27B 64% [42, 83], GPT-5.6-Sol 72% [53, 89] (preliminary); the first version (19 tasks pooled) is superseded within the folder.

## In the post?

Not in the post.

## Status

Current (figure 1 first version superseded by the corrected figure)

## Builds on

[2026-10-08_q2_free_checks_3](../2026-10-08_q2_free_checks_3/); [2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/); [2026-10-08_impactful_refit_bootstrap](../2026-10-08_impactful_refit_bootstrap/)

## Main scripts

- [`plot_figure1_old_new.py`](plot_figure1_old_new.py)
- [`plot_figure1.py`](plot_figure1.py)
- [`plot_figure2_forest.py`](plot_figure2_forest.py)
- [`plot_figure2_refit.py`](plot_figure2_refit.py)
- [`plot_pauc_meaning.py`](plot_pauc_meaning.py)
- [`find_placeholder_runs.py`](find_placeholder_runs.py)
- [`analyze_honest_rerun.py`](analyze_honest_rerun.py)

Result files the index names:

- [`figure1_old_new_qwen_2026-10-09_174208.json`](results/figure1_old_new_qwen_2026-10-09_174208.json)
- `figure1_old_new_qwen_2026-10-09_174208.png` (not copied: result file type not copied (.png))
- `placeholder_runs_2026-10-09_174122.json` (not copied: result left out to keep the release under 120 MB)
- [`figure1_old_new_qwen_2026-10-09_134724.json`](results/figure1_old_new_qwen_2026-10-09_134724.json)

## Files here

8 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-08_q2_figure_pack/` at commit `825327aa49c2a1cca3126f29c37ae1d1b949084e` of our internal repository. 36 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
