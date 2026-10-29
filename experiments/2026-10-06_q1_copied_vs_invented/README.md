# 2026-10-06_q1_copied_vs_invented

2026-10-06 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

How much of the real-vs-simulated separation comes from tool outputs the simulator invented versus outputs it copied from its reference honest run?

## What we did

Flagged every tool output of real attack epochs 1 and 2 and simulated attack (939 paired GPT-5.6-Sol cells) as exact, near or no match to the simulator reference run, then split out-of-fold TF-IDF text-classifier real-vs-simulated AUROC by group, against the two-real-runs comparison.

## What we found

README: per-output text-classifier AUROC 0.556 on exact copies, 0.847 on near copies, 0.820 on unmatched outputs (two real runs 0.494 to 0.554); real editor diffs always carry the line-number gutter, simulated ones in 7.6%.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-06_q1_text_baselines](../2026-10-06_q1_text_baselines/) (out-of-fold TF-IDF scores and staged table)

## Main scripts

- [`run_flag_outputs.py`](run_flag_outputs.py)
- [`analyze_split_auroc.py`](analyze_split_auroc.py)
- [`analyze_editor_gutter.py`](analyze_editor_gutter.py)

Result files the index names:

- [`flag_counts_2026-10-07_121021.json`](results/flag_counts_2026-10-07_121021.json)
- [`split_auroc_2026-10-06_133340.json`](results/split_auroc_2026-10-06_133340.json)
- [`editor_gutter_2026-10-06_133716.json`](results/editor_gutter_2026-10-06_133716.json)

## Files here

4 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-06_q1_copied_vs_invented/` at commit `b5f119618bea155709f28f3f66f3524096867fce` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
