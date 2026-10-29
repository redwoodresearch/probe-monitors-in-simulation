# 2026-10-11_q2_pooled_lean

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do all monitors (the 196-task command monitor, data transfer and destructive) catch slightly less in simulation, or is that noise?

## What we did

\$0 pre-registered pooled test on stored scores: pooled real-minus-simulated catch gap at 1% and 5% with a real-versus-real yardstick and a 3-point planted check.

## What we found

README: too insensitive to read; a 3-point plant was detected in 0 of 200 copies; descriptively the pooled gap at 1% is +4.4 points [+0.3, +8.4], driven by data transfer (+11.4), not distinguishable from two real runs.

## In the post?

Not in the post.

## Status

Exploratory (closed: too few)

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/); [2026-10-11_q2_margins_POSTHOC](../2026-10-11_q2_margins_POSTHOC/)

## Main scripts

- [`analyze_pooled_lean.py`](analyze_pooled_lean.py)

Result files the index names:

- [`pooled_lean_2026-10-11_142114.json`](results/pooled_lean_2026-10-11_142114.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_q2_pooled_lean/` at commit `745d128d777f5b583d2803d1c0851ec2cbabe4c3` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
