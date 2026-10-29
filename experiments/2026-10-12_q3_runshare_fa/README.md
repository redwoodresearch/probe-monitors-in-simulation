# 2026-10-12_q3_runshare_fa

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does the false-alarm workload match per run: what share of runs raise at least one false alarm, real vs simulated vs a second real run?

## What we did

\$0 reanalysis of stored per-call flags at the frozen 1% threshold (196-task command monitor, draw 0), with the real-vs-real comparison.

## What we found

19.4% first real, 16.2% simulated, 12.0% second real; simulated minus first real -3.1 points [-10.5, +4.2]; second minus first real -7.3 [-14.7, -0.5].

## In the post?

Not in the post.

## Status

Exploratory (post hoc)

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`analyze_runshare_fa.py`](analyze_runshare_fa.py)

Result files the index names:

- `runshare_fa_POSTHOC_2026-10-12_182016.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_runshare_fa/` at commit `3df378b592cfdc6398ad44c4fb1d31f8ee9b1057` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
