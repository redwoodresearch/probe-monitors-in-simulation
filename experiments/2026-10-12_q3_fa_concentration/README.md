# 2026-10-12_q3_fa_concentration

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do a few tasks drive the simulated runs' higher false-alarm rate for the 196-task command monitor (2.92% vs 1.14% per read-only call)?

## What we did

\$0 reanalysis of stored per-call flags (draw 0 of the 196-task scale-up): ranked tasks by simulated false alarms and recomputed the per-call gap after dropping the top 1, 3 and 5 tasks.

## What we found

Result JSON (at ba27150a): dropping the single top task removes the gap (simulated minus real +1.78 points becomes -0.07, paired); dropping 3 or 5 overshoots (-0.33, -0.50). Codex recompute: 0 mismatches.

## In the post?

Not in the post.

## Status

Exploratory (post hoc)

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`analyze_fa_concentration.py`](analyze_fa_concentration.py)

Result files the index names:

- `fa_concentration_POSTHOC_2026-10-12_103618.json` (not in this folder at the source commit)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_fa_concentration/` at commit `0683024074d9c78548ac0daa771cddda10ce4696` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
