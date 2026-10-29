# 2026-10-12_q3_writeup_figures

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Can three figures from the 2026-10-12 write-up outline be regenerated from raw data and match the images in the write-up document?

## What we did

\$0 plot scripts that read values by key from pinned result files (the editor-output formatting-fix test on all 939 cells, the check of how often the detector flags runs from their first commands, and the whole-run attack detector trained on Qwen3.5-27B's own runs), compare them with the document's images, and render PNGs.

## What we found

Three figures rendered: reply vs command readout under the formatting fixes, first-command catch, whole-run detector real vs simulated; superseded renders on S3.

## In the post?

Not in the post (figures for the 10-12 write-up outline).

## Status

Current

## Builds on

[2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/); [2026-10-12_writeup_numbers](../2026-10-12_writeup_numbers/) (FIGURES.md)

## Main scripts

- [`plot_eb4bbb7c.py`](plot_eb4bbb7c.py)
- [`plot_6a9b8621.py`](plot_6a9b8621.py)
- [`plot_fig_555eb7a0.py`](plot_fig_555eb7a0.py)

Result files the index names:

- `fig_eb4bbb7c_2026-10-12_135029.png` (not copied: figure (not copied))
- `fig_555eb7a0_whole_run_detector_2026-10-12_135215.png` (not copied: figure (not copied))
- `fig_6a9b8621_first_command_2026-10-12_113134.png` (not copied: figure (not copied))

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q3_writeup_figures/` at commit `c89f55798c83826fd33ffe5b34e7c55a14d3fb6e` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
