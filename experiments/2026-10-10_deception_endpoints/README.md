# 2026-10-10_deception_endpoints

2026-10-10 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Free run-level deception test: does the frozen Apollo-style deception probe on Qwen3.5-27B (layer 31) score attack runs above honest runs from stored command endpoints, and simulated like real?

## What we did

Mean layer-31 endpoint score per run from stored command-endpoint activations of the 196-task runs, attack vs honest with held-out calibration, then simulated vs real.

## What we found

README: attack vs honest AUROC 0.825 [0.791, 0.858] and 0.834 [0.797, 0.869]; same-condition controls near 0.5. Only 8 of 154 attack runs alarmed; simulated minus real -0.017 and -0.175, equivalence not established (preliminary).

## In the post?

Not in the post (the post's deception probe is the published FIBS weights).

## Status

Exploratory

## Builds on

[2026-10-09_q2_deception_apollo](../2026-10-09_q2_deception_apollo/) (frozen probe); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (stored endpoints)

## Main scripts

- [`analyze_endpoints.py`](analyze_endpoints.py)
- [`analyze_sim.py`](analyze_sim.py)
- [`calibrate_heldout.py`](calibrate_heldout.py)

Result files the index names:

- `metrics.json` (not copied: result left out to keep the release under 120 MB)
- [`roster_validation.json`](results/roster_validation.json)
- [`sim_validation.json`](results/sim_validation.json)
- `archive_manifest.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-10_deception_endpoints/` at commit `3b5a777fea1a06f652481dce546f4cce744443a4` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
