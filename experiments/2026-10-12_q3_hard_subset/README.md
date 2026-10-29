# 2026-10-12_q3_hard_subset

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does the acting set-up's command probe agree between real and simulated runs only on easy commands, ones a text classifier also catches?

## What we did

\$0, stored scores: restricted the 196-task command monitor's catch to impactful calls the command-text TF-IDF misses at its frozen 1% threshold and compared first real, simulated and second real catch with task resampling, plants and nulls.

## What we found

README: on hard calls simulated minus real -3.1 points [-8.5, +2.2], second real minus real -4.7 [-10.1, +0.2]; within the pre-set plus or minus 14-point margin, so agreement holds on hard commands.

## In the post?

Not in the post.

## Status

Exploratory (post hoc)

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`analyze_hard_subset.py`](analyze_hard_subset.py)
- [`count_hard_subset.py`](count_hard_subset.py)
- [`power_hard_subset.py`](power_hard_subset.py)

Result files the index names:

- `hard_subset_2026-10-12_005126.json` (not in this folder at the source commit)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_hard_subset/` at commit `0136606f79d7092dee9fd2298af377bb40cb8e83` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
