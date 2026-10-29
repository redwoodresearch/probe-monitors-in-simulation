# 2026-10-13_q3_sabotage_t2_realism

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

How well can a probe tell simulated from real runs at the readouts the sabotage alarm uses (command end, command tokens, tool outputs)?

## What we did

Realism probes on not-sabotage state-changing calls of 156 tasks (Qwen3.5-27B layer 30 own runs, and Qwen3-32B layer 63 cross-read), out of fold with inner-CV regularization, against a second-vs-first real-run control, with a v2 rerun after Codex's fixes. Modal, \$0.75 plus \$0.86.

## What we found

README (t2_realism_2026-10-13_180500.json): Qwen3.5 layer 30 simulated vs real AUROC 0.515 at command end and 0.508 at command last token ('no world signal'), 0.545 averaged command and 0.599 tool outputs ('world signal present'); Qwen3-32B layer 63 tool outputs 0.750 against a control of 0.526.

## In the post?

Not in the post (an early, small precursor of the simulation detector in Appendix E, which uses layer 48 on 1,313 tasks).

## Status

Exploratory

## Builds on

[2026-10-13_q3_sabotage_avg_repair](../2026-10-13_q3_sabotage_avg_repair/) (rows for the command readout averaged over tokens); [2026-10-13_q3_qwen3_crossread](../2026-10-13_q3_qwen3_crossread/)

## Main scripts

- [`run_t2_realism.py`](run_t2_realism.py)
- [`run_t2_realism_v2.py`](run_t2_realism_v2.py)

Result files the index names:

- `manifest_t2_s3_v1.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_t2_s3_v2.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_sabotage_t2_realism/` at commit `7bf8731ef21482e804178af880d8ff2708321e36` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
