# 2026-10-13_q2_watchers_v26

2026-10-13 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Can the Modal cost watchers tolerate a few failed billing polls (a laptop network drop stopped healthy apps) without losing their stop guarantee?

## What we did

Version 2.6 of the strict billing watcher and task guard: tolerate up to 3 failed polls while the bill is under 80% of the stop and the last good poll is under 300 s old; offline self-tests and four Codex review rounds.

## What we found

Self-tests pass ('ALL PASS'); deprecated on 2026-10-14 after its stale-to-stop rule killed a healthy server; replaced by [2026-10-14_q2_watchers_v27](../2026-10-14_q2_watchers_v27/).

## In the post?

Not in the post.

## Status

Superseded by [2026-10-14_q2_watchers_v27](../2026-10-14_q2_watchers_v27/)

## Builds on

Nothing earlier in this index.

## Main scripts

- [`watch_billing_strict_v26.py`](watch_billing_strict_v26.py)
- [`guard_tasks_v26.py`](guard_tasks_v26.py)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q2_watchers_v26/` at commit `2b7aa7c0b16a04024affcabfd15e01750153f90f` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
