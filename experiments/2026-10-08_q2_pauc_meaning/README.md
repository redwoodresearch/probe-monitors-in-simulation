# 2026-10-08_q2_pauc_meaning

2026-10-08 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

What does a drop in standardized partial AUROC (pAUC) mean in impactful calls caught at a 5% false-alarm operating point?

## What we did

Lowered the off-policy Qwen3.5-27B impactful-command probe's saved scores by hypothetical amounts (constant shift or dilution with read-only scores) and reported call and run catch, for a meeting; predictions committed first.

## What we found

A 0.03 drop costs 3.4 to 6.3 points of call catch and 0.5-0.7 points of run catch; hypothetical perturbations of one fixed score set, without intervals.

## In the post?

Not in the post.

## Status

Current (interpretation aid)

## Builds on

[2026-10-08_impactful_call_2x2](../2026-10-08_impactful_call_2x2/)

## Main scripts

- [`analyze_pauc_meaning.py`](analyze_pauc_meaning.py)

Result files the index names:

- [`pauc_meaning_2026-10-08_175417.json`](results/pauc_meaning_2026-10-08_175417.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-08_q2_pauc_meaning/` at commit `f4b4c9833d4d14b0256504b655577016ebc76c7d` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
