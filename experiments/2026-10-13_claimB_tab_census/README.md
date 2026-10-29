# 2026-10-13_claimB_tab_census

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Do simulated text_editor replies keep literal tabs where the real tool expands them (a formatting giveaway the fixes missed), and do any of the nine post hoc simulator formatting fixes touch tabs?

## What we did

Script census on Modal CPU of literal tabs in text_editor replies per arm (real first and second attack runs, real honest by batch, simulated attack and honest, fully fixed simulated), split by Go files vs other, then joins to probe scores, a real-honest Go split, and edit-reply format and inspect_ai version counts. About \$0.04.

## What we found

tab_census_POSTHOC_2026-10-13_173209.json: among runs with a Go-file reply, 69 of 77 simulated attack runs (89.6%) contain a literal tab, against 0 of 86 first real, 3 of 86 second real and 2 of 82 real honest runs; the full fix set leaves every tab unchanged (all transitions 0->0 or 1->1). This fed two later tab-formatting fixes in [2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_claimB_read10](../2026-10-13_claimB_read10/) (found the tabs); [2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/)

## Main scripts

- [`run_tab_census_modal.py`](run_tab_census_modal.py)
- [`run_tab_scores_modal.py`](run_tab_scores_modal.py)
- [`run_tab_real_modal.py`](run_tab_real_modal.py)
- [`run_edit_formats_modal.py`](run_edit_formats_modal.py)

Result files the index names:

- [`tab_census_POSTHOC_2026-10-13_173209.json`](results/tab_census_POSTHOC_2026-10-13_173209.json)
- [`tab_scores_POSTHOC_2026-10-13_174336.json`](results/tab_scores_POSTHOC_2026-10-13_174336.json)
- [`tab_real_POSTHOC_2026-10-13_175057.json`](results/tab_real_POSTHOC_2026-10-13_175057.json)
- [`edit_formats_POSTHOC_2026-10-13_182547.json`](results/edit_formats_POSTHOC_2026-10-13_182547.json)
- `eval_versions_POSTHOC_2026-10-13_180348.json` (not copied: result left out to keep the release under 120 MB)

## Files here

4 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-13_claimB_tab_census/` at commit `89daedc881f6451b70bd853d441e4e457df65f6d` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
