# 2026-10-08_q2_rewrite_audit

2026-10-08 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Are the staged text rewrites (swap arms, the 117-run extension, the surface-edit sweep) clean, changing only the targeted outputs?

## What we did

Diffed every item of eight staged rewrite runs against its original in the base dump on Modal CPU, with a regeneration check; predictions committed first.

## What we found

Only targeted outputs change, with one rewrite bug (about 0.1% of rewritten find-and-replace outputs numbered twice); regenerated messages match for all 1,117 items.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-07_q2_rendering_swap](../2026-10-07_q2_rendering_swap/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/); [2026-10-08_q2_surface_edit_sweep](../2026-10-08_q2_surface_edit_sweep/)

## Main scripts

- [`run_rewrite_audit.py`](run_rewrite_audit.py)

Result files the index names:

- `audit_2026-10-08_152106.json` (not in this folder at the source commit)
- `manifest_v6.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_q2_rewrite_audit/` at commit `02129bba87a153826abb50cdbba24dba6241d636` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
