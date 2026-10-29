# 2026-10-11_sim_toollist_swap

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Row 4: does the reader's view of the different tool definitions served to simulated runs cause the 196-task command monitor's higher false alarms on simulated runs (2.92% vs 1.14%)?

## What we did

Spliced the real tool-definition lines into each simulated run's system turn at the token level (and the reverse for real runs), re-extracted Qwen3.5-27B activations on Modal and rescored with the frozen monitor.

## What we found

README: no; giving simulated runs the real definitions moved false alarms by +0.015 points [-0.043, +0.098], ruling out explaining half the gap; the mirror test did not raise real false alarms; the definitions differ only in two schema fields.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (196-task monitor)

## Main scripts

- [`diff_tool_lists.py`](diff_tool_lists.py)
- [`build_tool_lines.py`](build_tool_lines.py)
- [`param_paths.py`](param_paths.py)
- [`check_splice_local.py`](check_splice_local.py)
- [`stage_toolswap.py`](stage_toolswap.py)
- [`score_toolswap.py`](score_toolswap.py)
- [`analyze_toolswap.py`](analyze_toolswap.py)
- [`check_all_tasks.py`](check_all_tasks.py)

Result files the index names:

- [`analysis_2026-10-11_203858.json`](results/analysis_2026-10-11_203858.json)
- `gates_2026-10-11_203852.json` (not copied: result left out to keep the release under 120 MB)
- `staging_2026-10-11_200321.json` (not copied: result left out to keep the release under 120 MB)
- `tool_list_fingerprints_2026-10-11.json` (not copied: result left out to keep the release under 120 MB)
- [`real_vs_sim_tool_pairs_2026-10-11.json`](results/real_vs_sim_tool_pairs_2026-10-11.json)

## Files here

8 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-11_sim_toollist_swap/` at commit `0a285659d66502d40342ffbeb5c928aae4e29108` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
