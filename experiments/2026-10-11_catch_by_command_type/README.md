# 2026-10-11_catch_by_command_type

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does the 196-task impactful-command probe's equal overall catch in real and simulated attack runs hide differences between command types, because simulated runs carry a different mix of types (Simpson's paradox)?

## What we did

On stored flags of the Qwen3.5-27B on-policy impactful-command probe (196 tasks, draw 0, frozen 1% threshold), compared real run 1 and simulated catch within each command type (tool and parameter name) and after reweighting both to one type mix, with real run 2 as a yardstick and a planted check; laptop CPU, no command text read.

## What we found

README: overall real minus simulated +0.46 points [-1.18, +2.20], reweighted +0.43 [-1.27, +2.21], within the 10-point margin; within types only bash:command shows +4.07 [-1.48, +10.02]; the planted check was detected; intervals hold fits and thresholds fixed, so they understate uncertainty.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (196-task scores, draw 0)

## Main scripts

- [`analyze_catch_by_type.py`](analyze_catch_by_type.py)

Result files the index names:

- `catch_by_type_2026-10-11_193855.json` (not copied: result left out to keep the release under 120 MB)
- `catch_by_type_2026-10-11_193855.png` (not copied: result file type not copied (.png))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_catch_by_command_type/` at commit `f84c079af271935f547819bf5a7a182dd051ba55` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
