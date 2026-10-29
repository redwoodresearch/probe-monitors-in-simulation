# 2026-10-13_q3_matchedfa_designs

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

When each world is calibrated on its own honest runs (matched false alarms), the edited-simulated minus second-real comparison could not be read; is there a design that reads it from the stored scores without new data?

## What we did

Compared four designs on stored scores for Qwen3-32B and Olmo-3.1-32B tool-output detectors (GPT-5.6-Sol runs): the as-run 5% catch with about 87 calibration runs per fold, pooled calibration of about 174 with each task left out, a threshold-free own-world AUROC, and a partial AUROC up to 10% false alarms, each gated on detecting the same planted shift. \$0, local.

## What we found

README (Codex recomputed): only the threshold-free own-world AUROC detects its plant for both models; under it edited minus real second run is -0.0109 [-0.0242, +0.0024] (Qwen3-32B) and +0.0018 [-0.0140, +0.0174] (Olmo), 'no difference detected at this design's sensitivity'; every catch-rate design misses the plant.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_reader_matched_fa](../2026-10-13_q3_reader_matched_fa/) (as-run matched false alarms)

## Main scripts

- none committed in git (scripts listed in manifest_script_s3_v1.json and manifest_script_s3_v2.json, on S3)

Result files the index names:

- `manifest_designs_s3_v1.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_designs_s3_v2.json` (not copied: result left out to keep the release under 120 MB)

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_matchedfa_designs/` at commit `3cf85860ee7aa2c16e764a0795c54f9743c5f4b6` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
