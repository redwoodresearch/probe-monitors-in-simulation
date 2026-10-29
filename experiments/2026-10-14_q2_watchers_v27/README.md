# 2026-10-14_q2_watchers_v27

2026-10-14 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Make the laptop billing watcher stop an app only on a fresh bill at or over its cap, after v2.6b killed a healthy server during a network outage.

## What we did

Wrote billing watcher v2.7 and a task guard, with a Modal-run test suite and a Codex review.

## What we found

Test suite ALL PASS (test_output_v27_2026-10-14_161300.txt).

## In the post?

Not in the post.

## Status

Current

## Builds on

Nothing earlier in this index.

## Main scripts

- [`watch_billing_strict_v27.py`](watch_billing_strict_v27.py)
- [`guard_tasks_v27.py`](guard_tasks_v27.py)
- [`test_watchers_v27.py`](test_watchers_v27.py)
- [`run_tests_modal.py`](run_tests_modal.py)

Result files the index names:

- [`test_output_v27_2026-10-14_161032.txt`](results/test_output_v27_2026-10-14_161032.txt)
- [`test_output_v27_2026-10-14_161300.txt`](results/test_output_v27_2026-10-14_161300.txt)

## Files here

4 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-14_q2_watchers_v27/` at commit `1c30603c23298ad66ba3e9d8266f53b8e112018a` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
