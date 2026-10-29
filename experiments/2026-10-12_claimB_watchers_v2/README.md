# 2026-10-12_claimB_watchers_v2

2026-10-12 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Tooling: can laptop-side watchers reliably stop a Modal job on cost or time, and does stopping by app name or by app ID work?

## What we did

Cost watcher v2.5 (container-time estimate plus billing) and a strict billing watcher, with offline fail-closed tests and a live stop test on a dummy sleeping Modal app.

## What we found

Stopping a detached app by name failed; stopping by app ID with --yes worked within 15 s; without --yes it failed. The watchers were copied into later launches.

## In the post?

Not in the post.

## Status

Current (tooling)

## Builds on

Nothing earlier in this index.

## Main scripts

- [`watch_cost.py`](watch_cost.py)
- [`watch_billing_strict.py`](watch_billing_strict.py)
- [`test_failclosed.py`](test_failclosed.py)
- [`test_stop_live.py`](test_stop_live.py)
- [`dummy_sleep_modal.py`](dummy_sleep_modal.py)

Result files the index names:

- `stop_live_2026-10-13_024359.json` (not copied: strings over 300 characters)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_watchers_v2/` at commit `5c8a15c8e062a1d1a3250561356bc394a0764ebf` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
