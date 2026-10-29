# 2026-10-13_q3_e2_calsize

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Would buying more simulated honest runs make the edited-simulated minus second-real comparison at matched false alarms readable?

## What we did

Price-only sizing: binomially thinned the task weights of the simulated and real calibration sets inside the 2,000 as-run bootstrap draws (Qwen3-32B, layer 63 tool-output readout, GPT-5.6-Sol runs), fit width = sqrt(A + B/m) and extrapolated the planted check's half-width to larger calibration sets. \$0; nothing launched.

## What we found

README (calsize_POSTHOC_2026-10-13_011125.json, Codex-recomputed): the planted check's upper half-width falls from 8.70 points today to 6.89 at 2x (+437 runs, about \$3.0k) and 4.82 at 8x (about \$21k), with a floor of 3.90; readable needs under 2.98, so no finite purchase reaches it.

## In the post?

Not in the post.

## Status

Exploratory (decision: buy nothing)

## Builds on

[2026-10-13_q3_reader_matched_fa](../2026-10-13_q3_reader_matched_fa/) and the matched-false-alarm design (as-run full-cohort 5% result)

## Main scripts

- [`size_calibration.py`](size_calibration.py)

Result files the index names:

- `calsize_POSTHOC_2026-10-13_011125.json` (not in this folder at the source commit)
- `manifest_calsize_s3_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_e2_calsize/` at commit `72c1b387fdab5785f8df971949ed85c4b4b834d1` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
