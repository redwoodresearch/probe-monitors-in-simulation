# 2026-10-17_step2_pilot_sold_equal

2026-10-17 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Show that the pilot runs' routing record matches an independent by-key reading before they are scored.

## What we did

Pilot copies of the full-length message builder and the equality checker, with a control (control_b1) comparing them with the originals.

## What we found

The last control result was DIFFERENT only at an adapter inventory-pin assertion, which was then scoped to the pilot path (NOTE_control_b1_rerun).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-15_step3_retrain](../2026-10-15_step3_retrain/) (original builders)

## Main scripts

- [`check_sold_equality.py`](check_sold_equality.py)
- [`build_cutfull_message.py`](build_cutfull_message.py)
- [`control_b1.py`](control_b1.py)
- [`check_fix_v2_subset.py`](check_fix_v2_subset.py)

Result files the index names:

- `control_b1_2026-10-17_062533.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-17_step2_pilot_sold_equal/` at commit `37111782cf993516cc7508600ae9075a0a3729c2` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
