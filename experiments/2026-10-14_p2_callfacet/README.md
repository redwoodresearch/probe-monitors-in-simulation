# 2026-10-14_p2_callfacet

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Do call-level monitors (the impactful-command monitors trained on real attacks and on simulated attacks, and the destructive/transfer "DT" monitors) score simulated attack calls like real ones?

## What we did

Scored stored five-fold monitor weights on the command readout of the 196-task replay layout (183-task cohort), per call and per run, at 1% and 5% thresholds from the second real honest arm and the third honest batch, with paired environment-stratified bootstraps, a false-alarm control on 20 simulated-honest tasks, and review-thread checks.

## What we found

The monitor trained on real attacks sits at the floor (catch about equal to the false-alarm rate): per-call sim − run 2 at 1% +0.06 points (95% −0.26 to +0.37); DT destructive per-run sim − run 2 +7.20 (95% −0.68 to +15.20), labelled a cross-launch comparison (callfacet_stats printout and labels file).

## In the post?

Not in the post.

## Status

Superseded by [2026-10-14_p2_dt_scaleup](../2026-10-14_p2_dt_scaleup/) (destructive-action and data-transfer monitors scored in the reader layout used for the scale-up)

## Builds on

[2026-10-14_priority1_spec_09](../2026-10-14_priority1_spec_09/) (spec appends)

## Main scripts

- [`run_callfacet_score.py`](run_callfacet_score.py)
- [`analyze_callfacet.py`](analyze_callfacet.py)
- [`check_rows.py`](check_rows.py)
- [`run_review_checks.py`](run_review_checks.py)

Result files the index names:

- [`gpu_tally_billing_2026-10-14.json`](results/gpu_tally_billing_2026-10-14.json)

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_p2_callfacet/` at commit `77078c1d08294c0a6bc601d45e344abde04b9798` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
