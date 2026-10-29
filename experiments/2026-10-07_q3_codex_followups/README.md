# 2026-10-07_q3_codex_followups

2026-10-07 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Cheap follow-ups from an AI reviewer's (Codex, GPT-6-Astra) critique of how the probe results were framed: is the first-phase attack detector's real-versus-simulated catch gap at tool outputs caused by the input text or by training, does the detector single out the sabotage call, and does a detector of attempted attacks behave alike on real and simulated runs?

## What we did

Ran follow-ups on the first-phase attack detector (a linear probe on Qwen3-32B activations at the end of each tool call, trained on GPT-5.6-Sol's attack and honest runs) using saved scores plus small Modal jobs: a 2x2 crossing the text-cleaning level the probe was trained on with the level it reads, an attempted-versus-declined attack detector with a joint table, 20 hand-labelled sabotage runs, learning curves, and a read of the frozen detector on Qwen3.5-27B's own 5 attack and 4 honest runs.

## What we found

Changing only the input text moves the tool-output gap by +25.9 [+8.4, +36.6], mostly by lowering catch on real runs. Sabotage calls ranked above benign ones at most 0.59. On Qwen3.5-27B's own runs honest runs flagged as often as attacks (2 of 4 vs 2 of 5).

## In the post?

Not in the post.

## Status

Current (exploratory)

## Builds on

[2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/)

## Main scripts

- [`run_attempt_detector.py`](run_attempt_detector.py)
- [`run_k6_learning_curves.py`](run_k6_learning_curves.py)
- [`run_read_phase2.py`](run_read_phase2.py)
- [`run_read_phase2_honest.py`](run_read_phase2_honest.py)
- [`analyze_k8_crossing.py`](analyze_k8_crossing.py)
- [`analyze_k5_detector.py`](analyze_k5_detector.py)
- [`analyze_k9_joint.py`](analyze_k9_joint.py)
- [`label_sabotage_dump.py`](label_sabotage_dump.py)

Result files the index names:

- `k8_crossing_output_end_2026-10-07_170538.json` (not copied: result left out to keep the release under 120 MB)
- [`attempt_detector_analysis_2026-10-08_011725__2026-10-08_012837.json`](results/attempt_detector_analysis_2026-10-08_011725__2026-10-08_012837.json)
- [`k9_joint_2026-10-08_020255.json`](results/k9_joint_2026-10-08_020255.json)
- [`k5_detector_2026-10-08_014434.json`](results/k5_detector_2026-10-08_014434.json)
- `phase2_honest_vs_attack_2026-10-08_000430.json` (not copied: result left out to keep the release under 120 MB)

## Files here

28 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-07_q3_codex_followups/` at commit `f2bc56b518700fcac8a40a7b77c363ca5aaff2f7` of our internal repository. 54 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
