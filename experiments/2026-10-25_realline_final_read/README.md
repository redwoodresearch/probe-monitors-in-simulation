# 2026-10-25_realline_final_read

2026-10-25 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

Is the slightly lower simulated catch caused by a higher simulated threshold rather than lower simulated attack scores, i.e. what is simulated catch against the real threshold?

## What we did

Read-only lookup, in the record of main pooled results at 1% and 5% alarm levels, of the stored real-threshold gap and its intervals for the six probes (and the Appendix D extra probe), six-probe mean intervals from stored draws, a clean-room half, and extraction of the per-draw arrays for the post's reproducibility folder; Modal CPU.

## What we found

The post's footnote: against the real threshold, simulated minus real catch -0.9 points on commands (95% -2.1 to +0.6) and -0.6 on tool outputs (-1.9 to +0.7); boot read 54/54 gates passed; clean-room half EXACT for both readouts (spend rows).

## In the post?

In the post: Main text "Qwen3.5's activations still gave the simulation away" (real-threshold check) and its footnote; Appendix C's real-threshold result

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) main record; [2026-10-16_cleanroom_pooled_repro](../2026-10-16_cleanroom_pooled_repro/)

## Main scripts

- [`run_realline_read.py`](run_realline_read.py)
- [`run_realline_boot_read.py`](run_realline_boot_read.py)
- [`run_derived_cleanroom_half.py`](run_derived_cleanroom_half.py)
- [`run_extract_realline_draws.py`](run_extract_realline_draws.py)

Result files the index names:

- `realline_boot_read_2026-10-25_180356.json` (not copied: result left out to keep the release under 120 MB)
- `realline_final_read_2026-10-25_172046.json` (not copied: result left out to keep the release under 120 MB)
- `derived_cleanroom_half_2026-10-26_104607.json` (not copied: result left out to keep the release under 120 MB)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-25_realline_final_read/` at commit `5bcec4babd927b9b7fe67aae3ae28bccf0e560d0` of our internal repository. 21 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
