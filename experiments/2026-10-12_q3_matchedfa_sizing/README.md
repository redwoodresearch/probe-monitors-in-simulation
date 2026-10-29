# 2026-10-12_q3_matchedfa_sizing

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

How many more tasks would put the equal-false-alarm comparison's 95% interval inside plus or minus 10 points?

## What we did

\$0 local projection: width of the real minus simulated catch interval at 1% on random task subsets (95, 142, all 191 tasks) and a fitted width-vs-tasks slope; price only, not launched.

## What we found

README: current interval [-6.27, +10.61], width 16.88 points; width barely shrinks with tasks (fitted exponent 0.15 vs 0.5 for 1/sqrt(n)), which Codex says bears little weight; the project lead decides.

## In the post?

Not in the post.

## Status

Price only, not launched

## Builds on

The 196-task matched-false-alarm analysis; [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`sizing_matched_fa.py`](sizing_matched_fa.py)

Result files the index names:

- `sizing_2026-10-12_105100.json` (not in this folder at the source commit)
- `SIZING.md` (not copied: notes or README (internal working labels))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_matchedfa_sizing/` at commit `db53b6d78fc484ae7466ddbd2ebab70cd907ba42` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
