# 2026-10-11_q2_margins_POSTHOC

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Post hoc: against explicit margins with Holm correction and plant sweeps, do the earlier real-versus-simulated results (behaviour, destructive and transfer monitors, deception-probe effects) show agreement or a difference?

## What we did

\$0 re-analysis of stored results: interval-inside-margin tests at plus or minus 5 and 10 points (or 0.25 standard deviations) with Holm, real-versus-real beside each, and plant sweeps for the smallest detectable shift; one figure per question.

## What we found

README: the destructive and transfer monitors both agree within plus or minus 10 points at 5% after Holm, not at 1% (transfer gap +6.8 [-1.0, +12.5]); for deception-probe effects only the run-level simulated-vs-real deception comparison falls inside the margin (-0.053 [-0.173, +0.067]); for behaviour no difference survives correction.

## In the post?

Not in the post.

## Status

Current (post hoc; threads closed)

## Builds on

The behaviour, destructive and transfer monitor, and deception-probe result folders (2026-10-06 to 2026-10-10)

## Main scripts

- [`analyze_q13_POSTHOC.py`](analyze_q13_POSTHOC.py)
- [`analyze_q25_POSTHOC.py`](analyze_q25_POSTHOC.py)
- [`analyze_q32_POSTHOC.py`](analyze_q32_POSTHOC.py)
- [`plot_q13_POSTHOC.py`](plot_q13_POSTHOC.py)
- [`plot_q25_POSTHOC.py`](plot_q25_POSTHOC.py)
- [`plot_q32_POSTHOC.py`](plot_q32_POSTHOC.py)

Result files the index names:

- [`q13_margins_POSTHOC_2026-10-11_003820.json`](results/q13_margins_POSTHOC_2026-10-11_003820.json)
- [`q25_margins_POSTHOC_2026-10-11_003031.json`](results/q25_margins_POSTHOC_2026-10-11_003031.json)
- [`q32_margins_POSTHOC_2026-10-11_003420.json`](results/q32_margins_POSTHOC_2026-10-11_003420.json)

## Files here

6 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_q2_margins_POSTHOC/` at commit `2321e208ba0d23716244a2f927db93bf60ca9e66` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
