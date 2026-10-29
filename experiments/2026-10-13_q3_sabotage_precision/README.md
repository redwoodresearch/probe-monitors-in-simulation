# 2026-10-13_q3_sabotage_precision

2026-10-13 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Can 16 held-out tasks measure the real-minus-simulated sabotage catch gap usefully, and how many labelled tasks would a +/-15-point interval need?

## What we did

Planning simulation (20,000 simulated studies per cell) of the 95% half-width and power for three sabotage detectors (panel labels, adjudicated labels, command endpoint) at the frozen 1% threshold, with costs per added task. \$0.

## What we found

README (precision_POSTHOC_A3 on S3): at 16 tasks the panel-label detector's half-width for real run 1 minus simulated is about +/-29 points; +/-15 needs 64 held-out tasks (48 new), about \$62 raw / \$80 with buffer; no feasible drop reaches 80% power at 16 tasks. This sized the scale-up.

## In the post?

Not in the post.

## Status

Exploratory (planning; fed [2026-10-13_q3_sabotage_scaleup](../2026-10-13_q3_sabotage_scaleup/))

## Builds on

Stage-1 sabotage gate ([2026-10-12_q3_sabotage_adjudication](../2026-10-12_q3_sabotage_adjudication/)); [2026-10-13_q3_sim_sabotage_ids](../2026-10-13_q3_sim_sabotage_ids/) (labels)

## Main scripts

- [`size_precision.py`](size_precision.py)
- [`summarize_precision.py`](summarize_precision.py)

Result files the index names:

- `manifest_precision_s3_v3.json` (not copied: strings over 300 characters)
- `precision_POSTHOC_A3_2026-10-13_125614.json` (not in this folder at the source commit)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_sabotage_precision/` at commit `73dc525c292a2077820c47c45f473453dffa63a6` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
