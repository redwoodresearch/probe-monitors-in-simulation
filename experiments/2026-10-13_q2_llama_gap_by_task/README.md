# 2026-10-13_q2_llama_gap_by_task

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Is Llama-3.3-70B's leftover 1% gap (second real run minus repaired simulated catch) concentrated in a few tasks or spread out?

## What we did

From stored nested scores of frozen probes (avg_vs_last conventions), split the gap into per-task contributions and left out the top 1, 3 and 5 tasks, for last-token and mean-pooled readouts, with a second-real-minus-first-real control. \$0.

## What we found

gap_by_task_POSTHOC_2026-10-13_115516.txt (via README): last-token 1% gap +6.34 [+3.11, +9.44], +4.96 without the top 5 tasks (kept share 0.78, 'spread out'); mean-pooled +4.10, kept 0.78 ('spread out'); controls +1.81 and -0.43.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q2_avg_vs_last](../2026-10-13_q2_avg_vs_last/); [2026-10-12_q2_llama_reader](../2026-10-12_q2_llama_reader/)

## Main scripts

- [`analyze_gap_by_task.py`](analyze_gap_by_task.py)

Result files the index names:

- `gap_by_task_POSTHOC_2026-10-13_115516.txt` (not copied: lines over 300 characters)
- `codex_numbers_gap_by_task_POSTHOC_2026-10-13.md` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `s3_manifest_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q2_llama_gap_by_task/` at commit `d8a2f76e3b548e4f0beeb2743c9c9c8b0e8a1d90` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
