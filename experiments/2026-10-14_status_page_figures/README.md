# 2026-10-14_status_page_figures

2026-10-14 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

(Tooling) Draw our internal status page's inline figures from committed result files.

## What we did

Scripts that read values by key with git show from committed records (versions 6 and 7 of the "error to add" analysis, the fully simulated comparison with thresholds set on simulated honest runs, the served-context check and the main first-figure file) and write SVG and HTML snippets for our internal status page.

## What we found

Figures drawn for our internal status page; no new numbers (README).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/); [2026-10-14_simhonest_a25_09](../2026-10-14_simhonest_a25_09/); [2026-10-14_served_check_09](../2026-10-14_served_check_09/)

## Main scripts

- [`extract_x.py`](extract_x.py)
- [`make_figs.py`](make_figs.py)
- [`snippet.py`](snippet.py)
- [`extract_detail_view.py`](extract_detail_view.py)
- [`make_detail_view.py`](make_detail_view.py)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_status_page_figures/` at commit `52ce61dd3e01560921ab45b9fdee3ef6f4b69c8e` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
