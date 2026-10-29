# 2026-10-14_xerr_scaleup_recheck

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does the scale-up "error to add" analysis (how many points a lab should add to its error bars when it measures probe catch in simulation) reproduce when recomputed from the raw scores by a harness written from the analysis plan alone?

## What we did

An independent checker built a harness without reading the original code, ran self-tests that reproduced versions 6 and 7 exactly, synthetic tests and a parse of the scale-up score folders, then compared the original final record key by key.

## What we found

Revision 6s: verdict "MATCH ON RECOMPUTED KEYS", 2,654 new keys compared with 0 mismatches; 104 numeric leaves not recomputed and listed (README).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_xerr_scaleup_09](../2026-10-14_xerr_scaleup_09/)

## Main scripts

- [`run_recheck_scaleup.py`](run_recheck_scaleup.py)
- [`make_key_map.py`](make_key_map.py)
- [`check_report_507_21j.py`](check_report_507_21j.py)
- [`analyze_label_maps.py`](analyze_label_maps.py)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_xerr_scaleup_recheck/` at commit `b7253730d977bcfecfb094755e8876a04b729216` of our internal repository. 118 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
