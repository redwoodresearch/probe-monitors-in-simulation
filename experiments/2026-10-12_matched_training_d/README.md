# 2026-10-12_matched_training_d

2026-10-12 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

If cue probes are trained on matched contrasts inside real honest transcripts (evaluation status and simulatedness varied independently), do they detect each cue in new wordings and held-out environments?

## What we did

Qwen3.5-27B on one H200: planted evaluation and simulation cue sentences into real honest runs (6 development, 2 selection, 2 never-used test environments), fit token-level probes at layers 2 to 63, selected on the selection set, tested with 20 per-task label-swap null refits; post hoc floor check and rescore with the shared system-prompt opening masked.

## What we found

Simulation cue (token max, layer 16) caught 116/116 with 0 false alarms, passes; evaluation cue (token max, layer 48) caught 101/116 with 8 false alarms (gate at most 2), fails; masking the shared opening changes no verdict.

## In the post?

Not in the post (the post's evaluation-awareness footnote uses the later [2026-10-14_p2_evalaware_nguyen](../2026-10-14_p2_evalaware_nguyen/) probe).

## Status

Exploratory

## Builds on

[2026-10-12_cue_pilot](../2026-10-12_cue_pilot/)

## Main scripts

- [`run_matched_d.py`](run_matched_d.py)
- [`analyze_matched_d.py`](analyze_matched_d.py)
- [`d_lib.py`](d_lib.py)
- [`run_POSTHOC_floor_check.py`](run_POSTHOC_floor_check.py)
- [`run_POSTHOC_prefix_rescore.py`](run_POSTHOC_prefix_rescore.py)
- [`analyze_POSTHOC_prefix.py`](analyze_POSTHOC_prefix.py)

Result files the index names:

- [`full_2026-10-12_172812.json`](results/full_2026-10-12_172812.json)
- `screen_2026-10-12_193321.json` (not copied: result left out to keep the release under 120 MB)
- `prefix_POSTHOC_screen_2026-10-13_012204.json` (not copied: result left out to keep the release under 120 MB)
- [`floor_check_POSTHOC_tokens_2026-10-12_204052.json`](results/floor_check_POSTHOC_tokens_2026-10-12_204052.json)

## Files here

7 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-12_matched_training_d/` at commit `d006589f2d1f996a000dbde52b064c504b3a39ce` of our internal repository. 19 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
