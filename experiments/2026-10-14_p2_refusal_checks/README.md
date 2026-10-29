# 2026-10-14_p2_refusal_checks

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Is the refusal direction's simulated-minus-real catch gap (the one probe whose interval excluded 0) a simulator effect, or explained by boring causes?

## What we did

Four cheap checks on stored records: a 14-row family Holm correction, attempt-label cohorts, run-mean vs record consistency and where in the run the top score sits, on Qwen3.5's 183-task pilot cohort.

## What we found

After a Holm correction over the 14-row family the refusal gap is not significant: +5.74 points at 5% (95% +0.28 to +11.02, p 0.040, Holm p 0.515) and +3.83 at 1% (Holm p 0.174) (refusal_checks_2026-10-14_153323.printout.txt).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_p2_refusal_harm](../2026-10-14_p2_refusal_harm/)

## Main scripts

- [`run_refusal_checks.py`](run_refusal_checks.py)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_refusal_checks/` at commit `566c019e3e1e76686358997a84d1166a958ff46c` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
