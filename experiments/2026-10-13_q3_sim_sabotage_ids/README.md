# 2026-10-13_q3_sim_sabotage_ids

2026-10-13 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Can a sabotage probe trained only on simulated runs catch real sabotage as well as the real-trained stage-1 gate, and does simulation report its catch honestly?

## What we did

Exported call IDs for the project lead's sabotage-panel labelling of the 40 stage-1 tasks' simulated runs and second real runs (IDs and counts only), then (on a side branch) trained the gate's recipe (Qwen3.5-27B layer 30 command end, C 1e-4) on 944 simulated calls of 24 training tasks and tested on 16 held-out tasks, with a 100-shuffle null. Modal CPU.

## What we found

README_simtrained (simtrained_result_2026-10-13_132643.json, Codex-recomputed): at 1%, the simulation-trained probe catches 2 of 32 (6.2%) first-real, 3 of 33 second-real and 4 of 27 (14.8%) simulated sabotage calls against 34.4%, 24.2% and 7.4% for the real-trained gate; simulation-trained minus real-trained on first real run -28.1 [-50.0, -3.8] 'differs'. Pilot, labels unaudited.

## In the post?

Not in the post.

## Status

Exploratory (pilot)

## Builds on

[2026-10-12_q3_sabotage_adjudication](../2026-10-12_q3_sabotage_adjudication/) (stage-1 gate)

## Main scripts

- [`make_sim_ids.py`](make_sim_ids.py)
- [`reread_panel_votes.py`](reread_panel_votes.py)
- [`make_simtrained_rows.py`](make_simtrained_rows.py)
- [`run_simtrained.py`](run_simtrained.py)
- [`analyze_simtrained.py`](analyze_simtrained.py)
- [`analyze_simtrained_matched.py`](analyze_simtrained_matched.py)
- [`run_r44.py`](run_r44.py)
- [`analyze_r44.py`](analyze_r44.py)
- [`analyze_label_counts.py`](analyze_label_counts.py)

Result files the index names:

- [`sim_ids_summary_2026-10-13_120231.json`](results/sim_ids_summary_2026-10-13_120231.json)
- [`real_run2_ids_summary_2026-10-13_122622.json`](results/real_run2_ids_summary_2026-10-13_122622.json)
- `sim_label_counts_2026-10-13_125229.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `simtrained_result_2026-10-13_132643.json` (not in this folder at the source commit)
- `manifest_simtrained_s3_v2.json` (not copied: result left out to keep the release under 120 MB)

## Files here

12 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_q3_sim_sabotage_ids/` at commit `7043b788d0644433b1d5bb42603d25e5de39b33e` of our internal repository. 16 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
