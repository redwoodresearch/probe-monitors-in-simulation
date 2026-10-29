# 2026-10-14_xerr_prelim_09

2026-10-14 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

Preliminary "error to add": how many points should a lab add to its error bars when it measures probe catch in simulation, from the probes scored so far on the pilot, with a per-task decomposition and a power projection?

## What we did

Computed from committed score records, in seven versions (tighter probe fits, refusal and harmfulness directions recomputed on H200 GPUs, rules for which probes are eligible, eligibility first cross-fitted and then frozen, and more probes), the pooled bias and an upper bound U on simulated-minus-real catch over the 90 pilot tasks held out for estimation, plus projections of U with more runs and tasks and sensitivity checks scoring individual commands and command ends.

## What we found

Error to add, version 6, reply readout at 1%: pooled bias −0.93 points (95% −5.74 to +3.89), error to add 0, approximate bootstrap upper limit 9.78 on 90 tasks; one projection variant put the upper limit near 4.9 points with 507 cells (the version 6 and projection_runs_v6 summaries).

## In the post?

Not in the post.

## Status

Superseded by [2026-10-14_xerr_scaleup_09](../2026-10-14_xerr_scaleup_09/) and then [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) (the final pooled analysis)

## Builds on

[2026-10-14_p2_fibs_pair](../2026-10-14_p2_fibs_pair/); [2026-10-14_p2_refusal_harm](../2026-10-14_p2_refusal_harm/); [2026-10-14_p2_highstakes](../2026-10-14_p2_highstakes/); [2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/)

## Main scripts

- [`run_xerr_prelim.py`](run_xerr_prelim.py)
- [`run_xerr_v2.py`](run_xerr_v2.py) … run_xerr_v7_expansion.py
- [`run_projection_v6.py`](run_projection_v6.py)
- [`run_projection_runs_v6.py`](run_projection_runs_v6.py)
- [`run_xerr_calllevel_dt.py`](run_xerr_calllevel_dt.py)

Result files the index names:

- `xerr_v6_2026-10-14_161122.json` (not copied: result file 1 MB or larger)
- `xerr_v7_phase1_2026-10-14_180820.json` (not copied: result left out to keep the release under 120 MB)
- `xerr_v7_phase2_2026-10-14_181336.json` (not copied: result file 1 MB or larger)
- `projection_runs_v6_2026-10-14_162944.json` (not copied: result left out to keep the release under 120 MB)

## Files here

13 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_xerr_prelim_09/` at commit `1d0b7f98537c939267e2e174b81cd183e6981c44` of our internal repository. 38 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
