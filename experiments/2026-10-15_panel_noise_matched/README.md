# 2026-10-15_panel_noise_matched

2026-10-15 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

Is the per-probe size of the simulated-vs-real gap bigger than the gap between two real runs, and would the comparison notice planted errors of known size?

## What we did

Defined the noise-matched gap (simulation gap minus rerun gap, absolute values, separate thresholds) on the pilot and pooled 455-task arrays; later held the final plant_rows job: lowered simulated attack scores to plant falls/rises of about 3, 5, 8 and 12 points on all 1,313 tasks and recorded which measures noticed them.

## What we found

The post's Table 6: the six-probe signed difference noticed a 3-point fall at 1% and 5%; single probes noticed 8-point changes in all six (falls) and all five (rises).

## In the post?

In the post: Appendix D Table 6 (planted errors); origin of the gap comparison in Appendix D / Figure 4

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) main record and PLANT record

## Main scripts

- [`run_plant_rows.py`](run_plant_rows.py)
- [`run_pooled.py`](run_pooled.py)
- [`check_regression.py`](check_regression.py)

Result files the index names:

- `panel_plant_rows_plant_rows_POOLED_FULL_routeC_2026-10-20_004247_2026-10-20_004833.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `routec_plant_rows_label_sidecar_2026-10-20_004247.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 1 result file the index names, they match in 3 script-and-commit pairs. Other result files, from earlier or side runs, give 14 script-and-commit pairs that match, 151 that differ and 114 where the script was not in the folder; we did not copy those versions.

## Files here

64 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_panel_noise_matched/` at commit `2d69ba99023164b82f6c8114eb3e942e3658233b` of our internal repository. 732 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
