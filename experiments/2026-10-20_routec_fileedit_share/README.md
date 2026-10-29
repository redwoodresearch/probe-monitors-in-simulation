# 2026-10-20_routec_fileedit_share

2026-10-20 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

What share of the commands the published probes score are file edits, in honest and attack runs, on the final inputs?

## What we did

Recounted command units by tool (shell vs file editor) on the final pinned inputs with the clean-room script (rev 3), keys and counts only; Modal CPU.

## What we found

The post's Appendix A: file edits about 17% of scored commands in honest runs and about 21% in attack runs.

## In the post?

In the post: Appendix A ("about 17%" honest, "about 21%" attack)

## Status

Current (labelled not independent, PRELIMINARY in its NOTE)

## Builds on

[2026-10-16_cleanroom_pooled_repro](../2026-10-16_cleanroom_pooled_repro/) (cunits_by_tool.py)

## Main scripts

- [`run_routec_fileedit_share.py`](run_routec_fileedit_share.py)
- [`launch_routec_fileedit_share.zsh`](launch_routec_fileedit_share.zsh)

Result files the index names:

- `cunits_by_tool_2026-10-20_044328.json` (not copied: result left out to keep the release under 120 MB)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_fileedit_share/` at commit `455a7ce531a271dd0dbd4e68efbf71a48b7d4bf4` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
