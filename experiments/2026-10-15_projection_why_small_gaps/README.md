# 2026-10-15_projection_why_small_gaps

2026-10-15 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Why can a detector tell simulated from real runs while probe catch gaps stay small: is the simulation shift mostly orthogonal to each probe's weights?

## What we did

Measured how much of the simulated-minus-real activation shift each probe's weight vector sees (projections across layers 19-63), refit the detector with saved weights and compared geometry; pilot cohort, Modal CPU/GPU (about \$1.07).

## What we found

Records written; values not read for this index. The post states the explanation as an untested guess.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_cmd_readout_simh_control](../2026-10-14_cmd_readout_simh_control/)

## Main scripts

- [`run_projection.py`](run_projection.py)
- [`run_detector_refit.py`](run_detector_refit.py)
- [`run_detector_geometry.py`](run_detector_geometry.py)

Result files the index names:

- `projection_why_2026-10-15_023002.json` (not copied: result left out to keep the release under 120 MB)
- `projection_why_addendum_2026-10-15_031049.json` (not copied: result left out to keep the release under 120 MB)
- `detector_geometry_2026-10-15_025937.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_projection_why_small_gaps/` at commit `0b8bdd4226fac670a6e9f152b1e5f73b34a4c038` of our internal repository. 19 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
