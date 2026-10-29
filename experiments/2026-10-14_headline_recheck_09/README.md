# 2026-10-14_headline_recheck_09

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does the "averaged off-to-on headline" (how much more a probe trained on Qwen3.5-27B's own runs catches than one trained on GPT-5.6-Sol's runs, both reading Qwen3.5's attacks) reproduce by a second route, and how robust is it?

## What we did

Rebuilt per-run scores on Modal CPU (Sol-trained probe from saved outer models, on-policy probe refit per fold, plus an sklearn refit as an independent fitter) on the 183-task pilot and recomputed catch and the difference with its own bootstrap; follow-ups ran 999-draw and 20,000-draw bootstraps, uncut transcripts, canonical layers and matched false-alarm variants.

## What we found

Reproduced exactly: on-policy probe 96.7% vs Sol-trained 56.8% catch at 1%, difference +39.9 points (95% +25.8 to +54.5; committed +26.1 to +54.5); with matched false alarms the difference is +27.9 (95% +7.6 to +39.3) (recheck_2026-10-14_015553 and cmatch_2026-10-14_060545 printouts).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/) (off-to-on headline, on a side branch)

## Main scripts

- [`recheck_headline.py`](recheck_headline.py)
- [`run_recheck_modal.py`](run_recheck_modal.py)
- [`bootstrap_999.py`](bootstrap_999.py)
- [`uncut_boot.py`](uncut_boot.py)
- [`canon_boot.py`](canon_boot.py)
- [`canon_matched.py`](canon_matched.py)
- [`lc_matched_fa.py`](lc_matched_fa.py)
- [`armrs_matched_fa.py`](armrs_matched_fa.py)

Result files the index names:

- [`recheck_2026-10-14_015553.json`](results/recheck_2026-10-14_015553.json)
- [`boot999_2026-10-14_022650.json`](results/boot999_2026-10-14_022650.json)
- [`uncut_boot_2026-10-14_025716.json`](results/uncut_boot_2026-10-14_025716.json)
- [`canonboot_2026-10-14_041007.json`](results/canonboot_2026-10-14_041007.json)
- [`cmatch_2026-10-14_060545.json`](results/cmatch_2026-10-14_060545.json)
- [`lcfa_2026-10-14_033527.json`](results/lcfa_2026-10-14_033527.json)
- [`armrsfa_2026-10-14_033853.json`](results/armrsfa_2026-10-14_033853.json)

## Files here

14 code and configuration files and 11 result files, copied byte for byte from `experiments/2026-10-14_headline_recheck_09/` at commit `e2de6673c7d56beaebae0fec311e1c655a85165d` of our internal repository. 15 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
