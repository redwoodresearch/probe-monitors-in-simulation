# 2026-10-14_q35_sol_avg_POSTHOC

2026-10-14 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Train and test the off-policy probe: Qwen3.5-27B reads GPT-5.6-Sol's runs, the probe averages over each tool output, and it is tested on Qwen3.5-27B's own runs against a probe trained on those runs (the on-policy reference). Over the sprint the folder became the main workspace for re-rendering transcripts correctly, fitting the off-policy probe and scoring every group of runs.

## What we did

Extracted Qwen3.5-27B activations over Sol's real and repaired simulated runs (all 64 layers, on H200 GPUs with per-shard reproduction checks), fit the probes trained on Sol's runs and the on-policy reference, built the list of Qwen3.5-27B's runs under the corrected rendering, then, over 74 amendments, re-fit the off-policy probe on full-length transcripts with its layers chosen again and scored every scale-up group of runs. The latest branch keeps only two scripts; the full folder is on a side branch.

## What we found

The input config of the final pooled analysis on all 1,313 tasks uses sola_uncut_reselect_fit_2026-10-16_030230_cutfull_491db0e7e2c9.json from this folder for the off-policy probe (per our table of the post's number sources). Early headline (README on a side branch): on-policy reference 98.4% vs Sol-trained 70.5% catch at 5% on 183 pilot tasks, difference +27.9 (95% +12.7 to +34.8).

## In the post?

In the post: Table 1, Table 2 and Appendix A (off-policy probe); Appendix D's extra mixed-training probe (simulated attack versus real honest runs; its full-length fit is pinned by the final input config)

## Status

Current

## Builds on

[2026-10-06_qwen35_replication](../2026-10-06_qwen35_replication/); [2026-10-13_recipe_transfer_POSTHOC](../2026-10-13_recipe_transfer_POSTHOC/)

## Main scripts

- [`run_extract_qwen35_avg.py`](run_extract_qwen35_avg.py)
- [`run_avg_transfer.py`](run_avg_transfer.py)
- [`run_asis_fits.py`](run_asis_fits.py)
- [`extract_q4b.py`](extract_q4b.py)
- [`stage_q35_repaired.py`](stage_q35_repaired.py)
- [`plot_p1_main.py`](plot_p1_main.py)

Result files the index names:

- `sola_uncut_reselect_fit_2026-10-16_030230_cutfull_491db0e7e2c9.json` (not in this folder at the source commit)
- `sola_uncut_reselect_training_keys_2026-10-16_030230_cutfull_491db0e7e2c9.json` (not in this folder at the source commit)
- `avg_transfer_avg_v1_2026-10-14_011347.json` (not in this folder at the source commit)
- `correctedb_manifest_v1.json` (not in this folder at the source commit)
- `sold_regress_2026-10-17_103840.json` (not in this folder at the source commit)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/199f50846/run_avg_transfer.py`](as_run/199f50846/run_avg_transfer.py) from commit `199f50846`: Trained the post's off-policy probe (Table 1, Table 2, Appendix A). The copy in this folder's snapshot is an older version of the script.
- [`as_run/7feaa2bf3/run_avg_transfer.py`](as_run/7feaa2bf3/run_avg_transfer.py) from commit `7feaa2bf3`: Trained Appendix D's simulated-attack probe (trained on simulated attack runs against real honest runs). The copy in this folder's snapshot is an older version of the script.
- [`as_run/2f1b932af/run_sola_uncut_score.py`](as_run/2f1b932af/run_sola_uncut_score.py) from commit `2f1b932af`: The off-policy probe's full-length scoring script at the commit that ran it. The folder's snapshot does not contain this script.
- [`as_run/9b2354740/run_extract_qwen35_avg.py`](as_run/9b2354740/run_extract_qwen35_avg.py) from commit `9b2354740`: The activation extraction for the off-policy probe's training runs, at the commit that added it as run (14 October). Two later commits (4beff9f6a, 0b0779aad) extended it for GPT-5.6-Sol's simulated runs; the latest of those is the copy in this folder. Git does not record which version produced the full-length activations the post used.
- [`as_run/687696e61/run_sold_cutfull_score.py`](as_run/687696e61/run_sold_cutfull_score.py) from commit `687696e61`: Scored the off-policy and simulated-attack probes on the 196 pilot tasks; release/pipeline/score.py cites it. The folder's snapshot does not contain this script.
- [`as_run/4d1eec572/run_avg_transfer.py`](as_run/4d1eec572/run_avg_transfer.py) from commit `4d1eec572`: The row reader for the on-policy probes' scoring, as release/pipeline/score.py cites it. The copy in this folder's snapshot comes from a different line of the history.

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran. The versions under `as_run/` come from our audit of which code produced the post's results.

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_q35_sol_avg_POSTHOC/` at commit `ea7c7d65cccca7229fc1260154c65477f2597a31` of our internal repository. Every file in that folder was copied. We also copied 4 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit. The 6 files under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
