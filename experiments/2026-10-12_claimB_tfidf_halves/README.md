# 2026-10-12_claimB_tfidf_halves

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does a raw-text term-frequency classifier (TF-IDF) also flag fixed-simulator runs more than second real runs in the halves design (train on one half of the environments and test on the other, in both directions) at the same 1-in-100 alarm?

## What we did

Gated on reproducing the archived TF-IDF baseline exactly, then ran the same TF-IDF pipeline through the halves design on Modal CPU, scoring simulated and second-real runs at the frozen 1% alarm with plant-2 checks and 2,000 task resamples.

## What we found

Codex fact-check: TF-IDF flags 24/939 simulated (2.56%) vs 7/939 second-real (0.75%), gap +1.81 points [+0.75, +4.22]; plant checks pass (probe: 51/939 vs 15/939).

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-11_claimB_halves](../2026-10-11_claimB_halves/)

## Main scripts

- [`run_tfidf_halves_modal.py`](run_tfidf_halves_modal.py)
- [`analyze_tfidf_halves.py`](analyze_tfidf_halves.py)

Result files the index names:

- `tfidf_halves_POSTHOC_2026-10-12_112423.json` (not in this folder at the source commit)
- `codex_factcheck_tfidf_halves_POSTHOC_2026-10-12_112423.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_tfidf_halves/` at commit `decf0833d125c382059d9c6e83c2b7f11dc3cb35` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
