# 2026-10-13_heldout_baselines_posthoc

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

In the held-out planting test, does planting still beat stronger rival forecasts: the gap at the nearest already-reported layer of the same reader, and the reader's mean reported gap?

## What we did

Built the two rival forecasts from committed files for 15 fixed-layer tool-output detectors (Qwen3-32B, Olmo-3.1-32B, Llama-3.3-70B reading GPT-5.6-Sol runs) and scored them with row 2's mean absolute error and joint task bootstrap (2,000 draws). \$0.

## What we found

Planting's average error 1.38 points against 4.52 for the nearest reported layer and 5.73 for the reader mean; rival minus planting +3.15 [+1.50, +3.96] and +4.35 [+2.52, +5.00]; for Llama-3.3-70B both rivals fall back to the historical baseline. Descriptive; cannot change row 2's verdict.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_plantslip_heldout](../2026-10-12_plantslip_heldout/) (row 2 test and its y)

## Main scripts

- [`make_b45.py`](make_b45.py)
- [`score_b45.py`](score_b45.py)

Result files the index names:

- `b45_forecasts_2026-10-13_005019.json` (not copied: result left out to keep the release under 120 MB)
- `b45_scores_2026-10-13_010013.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_heldout_baselines_posthoc/` at commit `b7b28efab8952f46d40e3d63a3411ffef3e77c30` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
