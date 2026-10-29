# 2026-10-10_panel_by_world

2026-10-10 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Are turns rated as suspecting a fake environment more common in simulated runs than in real ones? (the judge panel unblinded by arm)

## What we did

Joined the blind three-judge panel labels to the hidden world key by id, then compared the share of runs with a 'suspects fake' or 'notices something odd' turn between real and September-simulated attack runs, with task-paired bootstraps and a real-versus-real control.

## What we found

README: real attack 7 of 194 (3.6%) vs September simulated 3 of 191 (1.6%), +2.0 points [-0.6, +5.1]; two real runs differ by more, +3.1 points [+1.0, +5.7]; underpowered, no planted check.

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-10_blinded_environment_realism](../2026-10-10_blinded_environment_realism/); [2026-10-09_phase2_fake_env_judge](../2026-10-09_phase2_fake_env_judge/)

## Main scripts

- [`analyze_panel_by_world.py`](analyze_panel_by_world.py)

Result files the index names:

- `s3_manifest_v1.json` (not copied: strings over 300 characters)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_panel_by_world/` at commit `0475d56aca23da868beeb19e78b94c172eeb22b0` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
