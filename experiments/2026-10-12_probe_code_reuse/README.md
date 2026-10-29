# 2026-10-12_probe_code_reuse

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

For the planted-line probe, the evaluation-awareness probe and the is-this-a-test detector, which probe-training and scoring code does each reuse from the main monitor, and is anything reimplemented with bugs?

## What we did

Independent read-only Codex check of an agent's table of eight probe recipes against code at file and commit, plus a bug read of two fitters.

## What we found

Six separate fitting routines and many copied threshold rules; three code issues: the halves selection (train on one half of the environments and test on the other, in both directions) had no convergence check, the frozen deception-probe plant check compared the plant with itself (could never fail), one detector never checks its fit converged.

## In the post?

Not in the post.

## Status

Current (issues routed to [2026-10-12_claimB_convergence_check](../2026-10-12_claimB_convergence_check/) and others)

## Builds on

Nothing earlier in this index.

## Main scripts

- none (review only)

Result files the index names:

- `codex_review_2026-10-12_1159.md` (not copied: notes or README (internal working labels))
- `recipes_agent_summary.md` (not copied: notes or README (internal working labels))

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_probe_code_reuse/` at commit `08c6c18b99315c75d72b4b1ffd5ce528dd2fc5fa` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
