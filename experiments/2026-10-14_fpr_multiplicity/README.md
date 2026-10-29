# 2026-10-14_fpr_multiplicity

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Is the cluster of +5–6-point simulated-minus-real catch gaps seen near 1 alarm per 100 replies real, or chance across all the probe × readout × threshold settings of the false-positive-rate sweep?

## What we did

Sweep-wide multiplicity test (max-absolute, count-above-5-points, studentized and Holm tests against a task-bootstrap null) over a frozen family of five probes (Sol-trained, FIBS deception, harmfulness, refusal, high-stakes) at reply and command readouts on Qwen3.5-27B's 183-task pilot cohort, with a positive control; follow-ups split by real run and dry runs of a 507-task version.

## What we found

Simulated minus real over 250 settings: max |gap| 12.57 points (p 0.026), 41 settings at 5 points or more (p 0.017), 2 settings Holm-significant (FIBS command); run 2 minus run 1 also showed an excess count (50 settings, p 0.031) with max 11.48 points (p 0.217) (multiplicity_2026-10-14_163540.printout.txt).

## In the post?

Not in the post.

## Status

Exploratory (183-task pilot; the 507-task runner was only dry-run here)

## Builds on

[2026-10-14_fpr_sweep_unit](../2026-10-14_fpr_sweep_unit/); [2026-10-14_p2_fibs_pair](../2026-10-14_p2_fibs_pair/); [2026-10-14_p2_refusal_harm](../2026-10-14_p2_refusal_harm/); [2026-10-14_p2_highstakes](../2026-10-14_p2_highstakes/)

## Main scripts

- [`run_multiplicity.py`](run_multiplicity.py)
- [`run_multiplicity_followups.py`](run_multiplicity_followups.py)
- [`run_multiplicity_507.py`](run_multiplicity_507.py)

Result files the index names:

- `multiplicity_2026-10-14_163540.json` (not copied: result left out to keep the release under 120 MB)
- `followups_2026-10-14_164448.json` (not copied: result left out to keep the release under 120 MB)
- `multiplicity507_2026-10-14_172221.json` (not copied: result file 1 MB or larger)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_fpr_multiplicity/` at commit `b64028dd1a8b9ab28a775fb40a860901f4ffa294` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
