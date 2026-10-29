# 2026-10-08_q2_surface_edits

2026-10-08 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

How often does each simulator tell (line-number gutter, insert review note, timestamps, hex IDs) occur in real vs simulated runs, before the surface-edit sweep?

## What we did

One Modal CPU pass counting, per edit, the swap's 100 cells' real runs it would change, from the staged Qwen3-32B reader messages of all arms; predictions committed first.

## What we found

The line-number gutter appears on 99.9% of real run 1 diffs vs 7.7% simulated, and the insert review note on 96% of real inserts vs 1%; timestamps and hex IDs look similar (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-06_segment_end_extraction](../2026-10-06_segment_end_extraction/) (staged messages)

## Main scripts

- [`run_count_tells.py`](run_count_tells.py)

Result files the index names:

- `tell_counts_2026-10-08_154118.json` (not in this folder at the source commit)
- `manifest_v5.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-08_q2_surface_edits/` at commit `e7535ce930a0fb1ad6d8a8c1c8e410eee5399bc4` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
