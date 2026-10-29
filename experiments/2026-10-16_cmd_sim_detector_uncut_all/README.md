# 2026-10-16_cmd_sim_detector_uncut_all

2026-10-16 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Can a linear detector on full-length activations tell simulated from real attack runs across all 1,313 tasks, and does it fail on two real runs?

## What we did

Fit L2 logistic-regression detectors (layers 48 and 63; commands and tool outputs) for simulated vs real run 1 and for real run 2 vs run 1, five folds by main task, within-main label-shuffle null (100 draws), main-task bootstrap; Modal GPU (about \$7.42 first pass).

## What we found

Record attack_all_2026-10-16_042630 written; 3 of 40 fits uncertified, fixed in the follow-up refit of those 3 fits.

## In the post?

In the post: Figure 2; Table 7; Appendix E (record repaired by [2026-10-16_attack_refit3](../2026-10-16_attack_refit3/))

## Status

Superseded by [2026-10-16_attack_refit3](../2026-10-16_attack_refit3/) for the published values (same fits, 3 repaired)

## Builds on

[2026-10-16_cmd_sim_detector_uncut](../2026-10-16_cmd_sim_detector_uncut/); [2026-10-16_own_all_rs](../2026-10-16_own_all_rs/) fold list

## Main scripts

- [`run_attack_all.py`](run_attack_all.py)
- [`make_inputs_all.py`](make_inputs_all.py)
- [`run_simh_control.py`](run_simh_control.py)

Result files the index names:

- `attack_all_2026-10-16_042630.json` (not copied: result file 1 MB or larger)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/8ea4a6a5d/run_attack_all.py`](as_run/8ea4a6a5d/run_attack_all.py) from commit `8ea4a6a5d`: Made results/attack_all_2026-10-16_042630.json, the detector record behind Figure 2, Table 7 and Appendix E, which records commit 8ea4a6a5d. [2026-10-16_attack_refit3](../2026-10-16_attack_refit3/) refitted three of its fits with the same bytes (git blob a73f64d0b). The copy in this folder's snapshot (blob 94ca84faf) is a later edit that, per our methods audit, was held and never run.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 1 result file the index names, they match in 2 script-and-commit pairs, differ in 1. Other result files, from earlier or side runs, give 3 script-and-commit pairs that match; we did not copy those versions. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

11 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-16_cmd_sim_detector_uncut_all/` at commit `2399c49b351e57c06dfe8f4f1509a549ce4772d4` of our internal repository. 40 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 1 file under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
