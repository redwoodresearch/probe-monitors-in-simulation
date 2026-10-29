# 2026-10-13_g3_followups

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Where does Seed-OSS-36B's 13.63-point simulated-vs-real catch gap sit (by task and by tool-output type), how much do the main editor-output formatting repairs close, and do trailing newlines differ consistently between simulated and real outputs?

## What we did

With the frozen Seed-OSS-36B detector and bootstrap draws from the new-reader test: a \$0 breakdown of the gap by task and by output type (per-output scores computed on Modal CPU), Seed-OSS-36B scores on the simulated runs repaired with the editor-output fixes (paid), and a newline count across LinuxArena attack and honest runs, Qwen3.5-27B's own September runs and 91 matched sites.

## What we found

Only the newline census is committed: newline_census_table_POSTHOC_2026-10-13_144003.json says shell and view trailing newlines do not differ consistently (for example shell simulated minus real 'two or more newlines' -6.76 points on LinuxArena attack, -1.41 on September on-policy, +25.98 at the 91 sites; 'consistent' false). Breakdown and repaired-run results are on S3 (not read).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/); [2026-10-13_q2_llama_gap_by_task](../2026-10-13_q2_llama_gap_by_task/) and [2026-10-13_q2_llama_gap_where](../2026-10-13_q2_llama_gap_where/) (templates); the repaired runs of all 939 cells

## Main scripts

- [`run_newline_census.py`](run_newline_census.py)
- [`analyze_newline_census.py`](analyze_newline_census.py)
- [`analyze_rep_g3.py`](analyze_rep_g3.py)
- [`g3_score_rep.py`](g3_score_rep.py)
- [`run_newlines_g3.py`](run_newlines_g3.py)
- [`analyze_breakdown_amend1.py`](analyze_breakdown_amend1.py)

Result files the index names:

- [`newline_census_table_POSTHOC_2026-10-13_144003.json`](results/newline_census_table_POSTHOC_2026-10-13_144003.json)
- `newline_census_inputs_2026-10-13_143904.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_g3_followups/` at commit `a181b9710c6fb0fd7260217425a02473fafc3660` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
