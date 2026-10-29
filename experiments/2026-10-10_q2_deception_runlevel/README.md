# 2026-10-10_q2_deception_runlevel

2026-10-10 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does the frozen deception probe (Apollo's recipe on Qwen3.5-27B, layer 31) score whole attack-instructed runs above honest runs of the same task, and simulated attack runs like real ones?

## What we did

Under a plan frozen before scoring, run-level scores of the frozen probe on the 196-task runs: attack vs honest (171 tasks), then September simulated vs real attack runs, keys and numbers only.

## What we found

Attack runs scored +0.73 honest standard deviations higher (ranking 0.729) but only 5 of 171 crossed the alarm line; simulated vs real -0.053 [-0.173, +0.067]. Thread closed.

## In the post?

Not in the post (the post's deception probe is the published FIBS weights).

## Status

Closed

## Builds on

[2026-10-09_q2_deception_apollo](../2026-10-09_q2_deception_apollo/); [2026-10-09_q2_deception_commands](../2026-10-09_q2_deception_commands/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/)

## Main scripts

- [`run_check_and_score.py`](run_check_and_score.py)
- [`analyze_runlevel.py`](analyze_runlevel.py)

Result files the index names:

- [`runlevel_analysis_2026-10-10_215302.json`](results/runlevel_analysis_2026-10-10_215302.json)
- [`check_2026-10-10_215302.json`](results/check_2026-10-10_215302.json)

## Files here

2 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-10_q2_deception_runlevel/` at commit `d866a9474c14d16a0b7499e5fec71e545a1e8e45` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
