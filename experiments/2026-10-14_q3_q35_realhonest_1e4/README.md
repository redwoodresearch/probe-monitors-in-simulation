# 2026-10-14_q3_q35_realhonest_1e4

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Are per-reply and per-command false-alarm rates on Qwen3.5-27B's own real honest runs stable down to 1 in 10,000? That needs about 1,000 more real honest runs.

## What we did

Launched 980 real honest runs (five whole batches over the 196 pilot tasks) on a devbox with 6 H200 vLLM servers; a watcher incident stopped the first launch at gate 1, 74 affected runs were relaunched with the 699 never-started entries; then staged, format-checked and extracted on H200.

## What we found

Launch 1 stopped at gate 1 (205 usable of 229 ended) after the laptop billing watcher killed a healthy server (Amendment 8 to the experiment's written rules); after the relaunch, 958 runs form "honest_b_relaunch", an alternate calibration pool used only as a labelled sensitivity in the final pooled analysis, "never the lead's tau_B" (tau_B: the primary alarm threshold) ([2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/)/results/honest_b_relaunch_members_2026-10-16_032731.json).

## In the post?

Not in the post.

## Status

Current (sensitivity only)

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (harness and 196-task file)

## Main scripts

- [`launch_rh.sh`](launch_rh.sh)
- [`gate1.py`](gate1.py)
- [`select_relaunch.py`](select_relaunch.py)
- [`run_rh_stage.py`](run_rh_stage.py)
- [`extract_rh.py`](extract_rh.py)
- [`compare_repro.py`](compare_repro.py)
- [`post_run.sh`](post_run.sh)

Result files the index names:

- [`gate1_final_2026-10-14_152712.json`](results/gate1_final_2026-10-14_152712.json)
- [`relaunch_selection_2026-10-14.json`](results/relaunch_selection_2026-10-14.json)
- [`rh_stage_rh_2026-10-14_231559.json`](results/rh_stage_rh_2026-10-14_231559.json)
- [`extract_rh_rh1e4_extract_v1_a1_2026-10-14_232922.json`](results/extract_rh_rh1e4_extract_v1_a1_2026-10-14_232922.json)
- [`unusable_by_cause_2026-10-14_234651.json`](results/unusable_by_cause_2026-10-14_234651.json)

## Files here

31 code and configuration files and 9 result files, copied byte for byte from `experiments/2026-10-14_q3_q35_realhonest_1e4/` at commit `f0a371a297009b0b6b2c156a9ec04fece4635f0a` of our internal repository. 28 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
