# 2026-10-16_cmd_sim_detector_uncut

2026-10-16 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does the simulation detector give the same answer on whole runs as within the first 32,768 tokens?

## What we did

Reran the pilot-cohort detector design on full-length activations of every command and tool output (layers 48 and 63), with a 32k reproduction gate; Modal H100 (about \$4.93 billed).

## What we found

Full-length and 32k records written; values not read for this index.

## In the post?

Not in the post.

## Status

Superseded by [2026-10-16_cmd_sim_detector_uncut_all](../2026-10-16_cmd_sim_detector_uncut_all/) (all 1,313 tasks)

## Builds on

[2026-10-15_cmd_sim_detector](../2026-10-15_cmd_sim_detector/)

## Main scripts

- [`run_attack_uncut.py`](run_attack_uncut.py)
- [`run_honest_uncut.py`](run_honest_uncut.py)
- [`plot_detector_uncut.py`](plot_detector_uncut.py)

Result files the index names:

- `attack_uncut_2026-10-16_032244.json` (not copied: result left out to keep the release under 120 MB)
- `honest_uncut_2026-10-16_032243.json` (not copied: result left out to keep the release under 120 MB)
- `detector_uncut_auroc_2026-10-16_032244.png` (not copied: result file type not copied (.png))

## Files here

9 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_cmd_sim_detector_uncut/` at commit `9496c18e3bf749cd132b53f6019a4b4ec0deb1ef` of our internal repository. 27 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
