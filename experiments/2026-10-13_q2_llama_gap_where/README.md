# 2026-10-13_q2_llama_gap_where

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Where does Llama-3.3-70B's leftover gap sit: which kinds of tool outputs, byte-identical outputs or not, and is it simulation-specific?

## What we did

Per-output scores for Llama-3.3-70B's frozen probes (layers 60, 70) on second real, repaired-plus (simulated runs with the editor-output repairs plus the other frozen simulator-formatting fixes) and first real runs, split by output category, with post hoc mean-pooled, byte-identical-output and position/control analyses. Modal CPU for rows.

## What we found

README: registered reading at 5% 'disagreement', leading category 'ending newline'; at 1%, second-real-run catch minus catch on simulated runs with the additional formatting repairs +6.28 [+2.83, +9.16]; mean-pooled gap +4.10 led by the bash family; all-rows shift repaired-plus (simulated runs with the editor-output repairs plus the other frozen simulator-formatting fixes) -0.124 logits vs first real -0.026, difference -0.097 [-0.154, -0.041], 'simulation-specific (descriptive)'. Three result files were reshaped to pass a secret-scan hook (disclosed).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q2_llama_leftover](../2026-10-13_q2_llama_leftover/) (reading and scores); [2026-10-13_q2_avg_vs_last](../2026-10-13_q2_avg_vs_last/)

## Main scripts

- [`run_identical_POSTHOC.py`](run_identical_POSTHOC.py)
- [`run_rows_meanpool_POSTHOC.py`](run_rows_meanpool_POSTHOC.py)
- [`analyze_control_POSTHOC.py`](analyze_control_POSTHOC.py)
- [`analyze_control_diff_POSTHOC.py`](analyze_control_diff_POSTHOC.py)
- [`analyze_identical_POSTHOC.py`](analyze_identical_POSTHOC.py)
- [`analyze_meanpool_POSTHOC.py`](analyze_meanpool_POSTHOC.py) (registered run_rows.py and analyze_gap_where.py on S3)

Result files the index names:

- `gap_where_2026-10-13_112825_printout.txt` (not copied: lines over 300 characters)
- `meanpool_POSTHOC_2026-10-13_115302_printout.txt` (not copied: lines over 300 characters)
- [`identical_reading_POSTHOC_2026-10-13_115755_printout.txt`](results/identical_reading_POSTHOC_2026-10-13_115755_printout.txt)
- `control_POSTHOC_2026-10-13_131340_printout.txt` (not copied: lines over 300 characters)
- `control_diff_POSTHOC_2026-10-14_111205.json` (not copied: result left out to keep the release under 120 MB)

## Files here

6 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_q2_llama_gap_where/` at commit `0c33d7e688d3a8b6a8e9eac6f8ca4d505650c846` of our internal repository. 23 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
