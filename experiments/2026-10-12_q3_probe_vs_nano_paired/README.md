# 2026-10-12_q3_probe_vs_nano_paired

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

On the same 100 simulated attack runs, does the rendering edit move the reply-reading probe's catch more than the paper's nano monitor's detection?

## What we did

\$0 paired analysis of stored scores with 2,000 shared main-task resamples at each method's nominal 5% operating point.

## What we found

README: edit effect on the probe +20.5 points [+15.5, +25.3] vs nano +3.0 [-2.9, +8.9]; difference +17.5 [+9.9, +24.8].

## In the post?

Not in the post.

## Status

Exploratory (post hoc)

## Builds on

[2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/); the simulated control evaluations paper's monitor scores

## Main scripts

- [`analyze_probe_vs_nano.py`](analyze_probe_vs_nano.py)

Result files the index names:

- `probe_vs_nano_paired_POSTHOC_2026-10-12_010454.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_probe_vs_nano_paired/` at commit `c61759329d009549260cf8382297fca5e482db70` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
