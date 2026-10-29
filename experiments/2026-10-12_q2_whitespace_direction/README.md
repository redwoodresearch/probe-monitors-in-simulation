# 2026-10-12_q2_whitespace_direction

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Gate row 5: over the reply detector's 939 cells, which way do real and simulated tool outputs differ for each whitespace operation (line-final whitespace other than newlines, blank-line runs, trailing whitespace)?

## What we did

\$0 local count over second-real and simulated attack runs' tool outputs, applying the three whitespace operations from the fixed-simulator test and counting characters and outputs changed per run, by tool.

## What we found

Result JSON: line-final whitespace other than newlines: real has more in 611 of 939 cells; blank-line runs: real more in 431; trailing whitespace: simulated more in 675 (91.4% of simulated vs 61.4% of real outputs change).

## In the post?

Not in the post.

## Status

Exploratory (on a side branch only)

## Builds on

[2026-10-12_claimB_halves_number_shuffle](../2026-10-12_claimB_halves_number_shuffle/) (operation definitions); [2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/)

## Main scripts

- [`count_whitespace_direction.py`](count_whitespace_direction.py)

Result files the index names:

- [`whitespace_direction_2026-10-12_183627.json`](results/whitespace_direction_2026-10-12_183627.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-12_q2_whitespace_direction/` at commit `402ff0620cbc5b6486cb4e3cfbb4c81160eb69ed` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
