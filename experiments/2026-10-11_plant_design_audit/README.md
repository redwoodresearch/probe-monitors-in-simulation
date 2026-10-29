# 2026-10-11_plant_design_audit

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Which results of the off-policy probe work rest on the flawed planted-check design of the first-stage sabotage-probe gate (a plant sized in standard-deviation units, never calibrated for catch at the honest-run threshold)?

## What we did

\$0 read-only audit: classified each off-policy probe analysis's planted check by design (from its code) and counted detections from result files with a script.

## What we found

AUDIT.md: the stage-1 gate's plant was not detected at the real model choice and caught only 3.5% at the planted copy's choice; detection counts per design class in plant_detection_counts_2026-10-11_224833.json; the frozen-probe plants are in the design class of score-offset plants calibrated to AUROC 0.85, narrower than the full pass rule.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_sabotage_test](../2026-10-11_sabotage_test/) (stage-1 gate); [2026-10-11_frozen_nabil_probes](../2026-10-11_frozen_nabil_probes/)

## Main scripts

- [`count_plant_detection.py`](count_plant_detection.py)

Result files the index names:

- [`plant_detection_counts_2026-10-11_224833.json`](results/plant_detection_counts_2026-10-11_224833.json)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_plant_design_audit/` at commit `4fb1200e5136a3e88cd3f1583e41f0801a2ecc4d` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
