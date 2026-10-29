# 2026-10-14_xerr_scaleup_09

2026-10-14 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

On the scale-up cohort (the 196 pilot tasks plus 311 new tasks), does adding tasks tighten the "error to add" (how many points a lab should add to its error bars when it measures probe catch in simulation), and do the pilot and new tasks agree?

## What we did

Implemented the scale-up rule (probe eligibility frozen from the pilot, which cross-validation copy of each probe scores each new task, interim and final specs) and computed the error to add with its bias and upper bound per readout and alarm level, separately for pilot and new tasks, plus a release record and a report under the pre-registered rule for comparing and presenting the task groups.

## What we found

Reply readout at 5%, 362 estimation tasks: family bias +0.3 points (95% −2.3 to +2.8), error to add 2.8, approximate bootstrap upper limit 5.7, against a rerun yardstick of 1.6 / 5.1; per probe, Sol-trained +5.0 (95% +1.0 to +8.9) (report_507_21j_2026-10-15_095045.md).

## In the post?

Not in the post.

## Status

Superseded by [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) (the final pooled analysis on all 1,313 tasks)

## Builds on

[2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/); [2026-10-14_p2_family_expansion](../2026-10-14_p2_family_expansion/) (pass 2 scores); [2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/)

## Main scripts

- [`run_xerr_scaleup.py`](run_xerr_scaleup.py)
- [`make_interim_spec.py`](make_interim_spec.py)
- [`report_507_21j.py`](report_507_21j.py)
- [`check_repro_rev6s.py`](check_repro_rev6s.py)

Result files the index names:

- `xerr_scaleup_final_2026-10-15_013255.json` (not copied: result file 1 MB or larger)
- `report_507_21j_2026-10-15_095045.md` (not copied: lines over 300 characters)
- `RELEASE_507.md` (not copied: lines over 300 characters)

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_xerr_scaleup_09/` at commit `2bf70468dbd35ef74cfd43ac8634911848a0c4b0` of our internal repository. 83 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
