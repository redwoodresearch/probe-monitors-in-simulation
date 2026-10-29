# 2026-10-11_q2_tfidf_misses

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does the impactful-command probe agree between real and simulated runs only on easy commands, or also on impactful commands a text classifier (TF-IDF) misses?

## What we did

\$0 on stored labels and scores: restricted to impactful commands the TF-IDF misses at its frozen 1% threshold and compared the real-trained probe's catch in real and simulated runs, with an equivalence margin and a 10-point planted check.

## What we found

README: on hard commands the probe catches 27.3% real vs 27.7% simulated (-0.4 [-4.7, +3.7]), agreeing within plus or minus 13.5 points; plant detected 200 of 200; no two-real-runs yardstick.

## In the post?

Not in the post.

## Status

Exploratory (closed)

## Builds on

[2026-10-09_q2_tpr_at_1pct](../2026-10-09_q2_tpr_at_1pct/); [2026-10-08_q2_free_checks](../2026-10-08_q2_free_checks/)

## Main scripts

- [`analyze_tfidf_misses.py`](analyze_tfidf_misses.py)

Result files the index names:

- [`tfidf_misses_2026-10-11_144002.json`](results/tfidf_misses_2026-10-11_144002.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_q2_tfidf_misses/` at commit `90b70dc50e2902c846db2a1165567b6661152662` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
