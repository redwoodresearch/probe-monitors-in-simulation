# 2026-10-11_sabotage_monitor_sizing

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

How many hand-labelled sabotage commands would a plus or minus 10-point real-versus-simulated test of the 196-task command monitor's sabotage-command catch need?

## What we did

\$0 facts and Monte Carlo sizing from the monitor's stored draw-0 scores and the project lead's 60 hand-labelled keys (30 sabotage, 30 benign); keys and numbers only.

## What we found

README: the monitor flags 28 of 30 keyed sabotage commands (93.3%) but also all 30 benign ones (sabotage-vs-benign AUROC 0.46), so it flags them for being impactful; the sizing table gives tasks and commands needed per arm.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (196-task draw 0)

## Main scripts

- [`analyze_facts.py`](analyze_facts.py)
- [`run_sizing.py`](run_sizing.py)

Result files the index names:

- `facts_2026-10-11_153932.json` (not in this folder at the source commit)
- `sizing_2026-10-11_153952.json` (not in this folder at the source commit)
- `manifest_sizing_v1.json` (not copied: strings over 300 characters)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_sabotage_monitor_sizing/` at commit `5fef7e7a039fe8e53dbd60ce8dbd2211f0fe423b` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
