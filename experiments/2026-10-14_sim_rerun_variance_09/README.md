# 2026-10-14_sim_rerun_variance_09

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Is the simulator's run-to-run variance per task the same as the real runs', as the runs-per-task projection assumed?

## What we did

Used the extra purchase's second simulated attack epoch on 19 pilot tasks: per probe, readout and alarm level, compared within-task variance of simulated hits with that of real runs 1 and 2, then re-ran the runs-per-task projection with the measured variance.

## What we found

Pilot, n ≈ 19, underpowered: most cells "not contradicted at this n"; a few show the simulator's variance entirely above or below the real one (rerunvar_real_2026-10-14_181402.json, readings).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_q3_q35_simhonest_purchase](../2026-10-14_q3_q35_simhonest_purchase/) (the extra purchase); [2026-10-14_xerr_prelim_09](../2026-10-14_xerr_prelim_09/) (projection)

## Main scripts

- [`run_sim_rerunvar.py`](run_sim_rerunvar.py)
- [`run_projection_runs_measured.py`](run_projection_runs_measured.py)

Result files the index names:

- `rerunvar_real_2026-10-14_181402.json` (not copied: result left out to keep the release under 120 MB)
- `rerunvar_real_e2attempted_2026-10-14_182554.json` (not copied: result left out to keep the release under 120 MB)
- `projection_runs_measured_2026-10-14_181542.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_sim_rerun_variance_09/` at commit `6b85dd4710aa1010eee8736717a00a26d1059f03` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
