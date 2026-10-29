# 2026-10-13_g3_newline_control

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does the trailing newline on simulated text-editor confirmations (real ones have none) move Seed-OSS-36B's frozen attack detector, and through which route?

## What we did

Two edited versions of the 939 paired task cells: simulated runs with every trailing newline stripped from create, str_replace and insert outputs, and second real runs with one newline appended to each; Seed-OSS-36B re-read both, the frozen probes and 1% thresholds from the new-reader test scored them, and per-output shifts were split by position relative to the first edited output. Modal GPU, about \$57.5.

## What we found

README (from nl_reading_POSTHOC_2026-10-13_143414.json): stripping raises simulated catch 35.30 to 39.67 (+4.37 [3.51, 5.27]); adding lowers real catch 48.94 to 42.92 (-6.02 [-7.13, -4.95]); stripping closes 0.32 [0.24, 0.42] of the 13.63-point gap, and the effect acts through history (rows after the first edit move, growing with the number of edited outputs before them).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/)

## Main scripts

- [`score_nl.py`](score_nl.py)
- [`analyze_nl.py`](analyze_nl.py)
- [`analyze_render_nl.py`](analyze_render_nl.py)
- [`run_extract_g3_nl.patch`](run_extract_g3_nl.patch)
- [`launch/nl_launch.sh`](launch/nl_launch.sh)

Result files the index names:

- [`scores_nlstrip_2026-10-13_142236.json`](results/scores_nlstrip_2026-10-13_142236.json)
- [`scores_nladd_2026-10-13_143345.json`](results/scores_nladd_2026-10-13_143345.json)
- [`render_nlstrip_gate_2026-10-13_135255.json`](results/render_nlstrip_gate_2026-10-13_135255.json)
- [`render_nladd_gate_2026-10-13_135735.json`](results/render_nladd_gate_2026-10-13_135735.json)
- `s3_manifest_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

6 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-13_g3_newline_control/` at commit `caa701d4075fb85ddb7e86f6ca497da67d39cdf1` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
