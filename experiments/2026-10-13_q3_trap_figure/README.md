# 2026-10-13_q3_trap_figure

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Figure: how much does a probe trained on unrepaired simulated attacks vs real honest runs (the 'mixed' recipe) overstate catch, across five readers?

## What we did

Extracted plotted values by key from committed metrics files (sha256-checked) and plotted the mixed probe's simulated vs real catch beside a real-trained probe for Qwen3-32B, Qwen3.5-27B, Seed-OSS-36B, Olmo-3.1-32B and Llama-3.3-70B, on Modal CPU (about \$0.05).

## What we found

README: the mixed probe reports 81 to 94% catch on simulated attacks but catches 12 to 29% of real attacks (38 to 54% for a real-trained probe); overstatement 52 to 79 points at a 1% target and 36 to 72 at 5%. A figure of existing numbers; not in the post.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) and the reader folders' metrics files

## Main scripts

- [`extract_trap_numbers.py`](extract_trap_numbers.py)
- [`plot_trap.py`](plot_trap.py)

Result files the index names:

- `trap_figure_2026-10-13_230756.png` (not copied: figure (not copied))
- `trap_numbers.csv` (not copied: file type not copied (.csv))

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_trap_figure/` at commit `0c6ca212ac0e029caaf26b158282a2b7bec9cbd1` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
