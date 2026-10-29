# 2026-10-14_cmd_readout_simh_control

2026-10-14 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Do Qwen3.5-27B's activations at the command (end-of-call) readout carry a detectable simulator signature, or only the averaged reply readout?

## What we did

Trained linear detectors on stored Qwen3.5-27B activations (196-task pilot, corrected render, cut at 32,768 tokens) for simulated honest vs real honest runs, then (Amendments 1–3) simulated attack vs real run 1 against real run 2 vs run 1 as a noise floor, at reply and command readouts and layers 48 and 63, with within-task label-shuffle nulls and task-paired bootstraps; Amendment 3 restricted to attempted runs.

## What we found

Simulated vs real honest: reply L48 AUROC 0.615 (null p95 0.536), command L48 0.529 (null p95 0.543) (simh_control printout). Attack: reply L48 sim vs run 1 AUROC 0.796 vs 0.495 for run 2 vs run 1; command L63 0.594 vs 0.521; the pre-set rule was met at reply readouts but not at command readouts (report_a2_a3_2026-10-15.txt).

## In the post?

Not in the post.

## Status

Superseded by the all-task simulation detectors behind Appendix E (e.g. [2026-10-15_cmd_sim_detector](../2026-10-15_cmd_sim_detector/), [2026-10-17_routec_slot_readout](../2026-10-17_routec_slot_readout/))

## Builds on

[2026-10-14_q3_q35_simhonest_purchase](../2026-10-14_q3_q35_simhonest_purchase/) (simulated honest dumps); [2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/) (corrected-B manifest)

## Main scripts

- [`run_simh_control.py`](run_simh_control.py)
- [`run_controls_a1.py`](run_controls_a1.py)
- [`run_controls_a2.py`](run_controls_a2.py)
- [`run_controls_a3.py`](run_controls_a3.py)
- [`preflight_a3.py`](preflight_a3.py)
- [`report_a2_a3.py`](report_a2_a3.py)

Result files the index names:

- `simh_control_2026-10-14_232223.json` (not copied: result left out to keep the release under 120 MB)
- `controls_a1_2026-10-14_235656.json` (not copied: result file 1 MB or larger)
- `controls_a2_2026-10-15_004048.json` (not copied: result left out to keep the release under 120 MB)
- `controls_a3_2026-10-15_010644.json` (not copied: result left out to keep the release under 120 MB)
- `report_a2_a3_2026-10-15.txt` (not copied: lines over 300 characters)

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_cmd_readout_simh_control/` at commit `c663d0204ada19d18101c3455907ff775b417c25` of our internal repository. 29 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
