# 2026-10-11_claimB_swap_changed

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Counting only sites where the simulated reply differs from the real one, does one simulated reply carry at least half of the probe's gap (a forward test of the k = 1 post hoc split)?

## What we did

Pre-registered plan with power analysis and a score-blind state-compatibility audit (one tiny Modal job) of candidate swap pairs; closed before any reads were bought.

## What we found

PLAN.md: closed ('Close row 13'); the audit left 40 target pairs and 9 real-versus-real yardstick pairs, so the power model put the chance of passing the yardstick check at 0; no test was run.

## In the post?

Not in the post.

## Status

Void (closed before any read)

## Builds on

[2026-10-11_claimB_k_replies](../2026-10-11_claimB_k_replies/)

## Main scripts

- [`analyze_POWER_swap_changed.py`](analyze_POWER_swap_changed.py)
- [`analyze_POWER_stepfork.py`](analyze_POWER_stepfork.py)
- [`run_stepfork_audit_modal.py`](run_stepfork_audit_modal.py)

Result files the index names:

- `power_stepfork_2026-10-11_211112.json` (not copied: result left out to keep the release under 120 MB)
- `power_stepfork_audited_2026-10-11_224405.json` (not copied: result left out to keep the release under 120 MB)
- `results_manifest_v2.json` (not copied: JSON outside results/ and config/)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_swap_changed/` at commit `50bf69a2ef9f990a8b984e44ce9c552d4468b145` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
