# 2026-10-13_q3_sce_reply_probe

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Would a tool-output probe built the way a simulated control evaluation would build it (simulated attacks vs real honest runs) learn 'simulated = attack', look great in simulation and fail on real attacks; do the formatting repairs rescue it; and how do simulation-only and real-trained recipes compare?

## What we did

Qwen3-32B reading GPT-5.6-Sol runs, last-token tool-output readout at layers 48 and 63, with probes trained on real attack vs real honest runs (cell a), simulated attack vs simulated honest (cell b), simulated attack vs real honest (mixed, cell c), and repaired versions of b and c, on the 939-cell cohort with 100-permutation label-shuffle nulls, a planted cue and equal-count checks. Many amendments added count-matched real training, a token-averaged readout, Seed-OSS-36B, Olmo-3.1-32B and Llama-3.3-70B grids, and variants with all formatting fixes and with the file-view newline fix. Modal.

## What we found

README (S3 results, Codex recomputes agree): at 1% the mixed probe catches 84.4% (layer 48) and 93.9% (layer 63) of simulated attacks but 17.6% and 14.9% of real ones, overstating real catch by 66.8 [61.2, 70.2] and 78.9 [73.5, 81.2] points; repairs cut that by 35.7 and 29.3 points; the simulation-only probe overstates by 6.1 and 13.3 but catches fewer real attacks (20.3%, 24.8%) than the real-trained probe (36.9%, 38.1%).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

The attack detector ([2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/)); the repaired arm of all 939 cells; [2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/)

## Main scripts

- [`run_sce_fixfull.py`](run_sce_fixfull.py)
- [`run_sce_seed_variants.py`](run_sce_seed_variants.py)
- [`analyze_sce_probe.py`](analyze_sce_probe.py)
- [`analyze_sce_matched.py`](analyze_sce_matched.py)
- [`analyze_sce_fixfull.py`](analyze_sce_fixfull.py)
- [`analyze_sce_equal.py`](analyze_sce_equal.py)
- [`analyze_sce_grid.py`](amendment12_readers_grid/analyze_sce_grid.py)
- [`analyze_sce_seed.py`](analyze_sce_seed.py)
- [`analyze_sce_qwen35.py`](analyze_sce_qwen35.py)
- [`amendment12_readers_grid/run_sce_grid.py`](amendment12_readers_grid/run_sce_grid.py)

Result files the index names:

- `manifest_sce_results_v1.json` (not copied: result left out to keep the release under 120 MB)
- `sce_fixfull_metrics_POSTHOC_2026-10-13_153447.json` (not in this folder at the source commit)
- `simhonest_fullfix_c_full_runscores_2026-10-13_152618.csv` (not in this folder at the source commit)
- `manifest_sce_grid_results_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

51 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_sce_reply_probe/` at commit `888f2a89abdec9eb1bcde1483a088fa74b4bf734` of our internal repository. 60 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
