# 2026-10-14_q3_q35_simhonest_purchase

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Buy simulated honest runs of Qwen3.5-27B (September simulator) on the 196 pilot tasks, to train a probe on Qwen3.5's own simulated runs and to set simulated alarm thresholds; plus an extra purchase of a second simulated attack epoch on 20 tasks.

## What we did

Devbox launches with the reviewed harness and gates (settings equality, approach hashes, H200-only extraction with per-shard reproduction): a 20-task pilot, the extra purchase (20 tasks), a continuation in two batches, and a rerun of task 43; staged with the corrected render and extracted.

## What we found

Pilot 20 of 20 usable, \$19.01 per run, \$396.08 all-in; the extra purchase 19 of 20 usable (one run lost to the watcher); continuation batch 1: 145 of 150 usable at the time of that snapshot (four runs were still running; an interim count) (Amendments 10 and 16 to the experiment's written rules). The input config of the final pooled analysis on all 1,313 tasks lists simh_q4f_v1_pilot and full_b1/b2 as its simulated-honest stagings.

## In the post?

In the post: Appendix A (simulated honest runs that set the simulated threshold for the oldest stratum)

## Status

Current

## Builds on

[2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/) (corrected staging path)

## Main scripts

- [`check_launch_tasks.py`](check_launch_tasks.py)
- [`check_sim_settings.py`](check_sim_settings.py)
- [`check_approach_hashes.py`](check_approach_hashes.py)
- [`check_compose_race.py`](check_compose_race.py)
- [`check_tool_defs.py`](check_tool_defs.py)
- [`allin_stop.sh`](allin_stop.sh)

Result files the index names:

- `simh_counted_keys_primary_v3.json` (not copied: strings over 300 characters)
- `simh_counted_keys_pilot_v3.json` (not copied: strings over 300 characters)
- `simh_attack2_pairing.json` (not copied: strings over 300 characters)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. Other result files, from earlier or side runs, give 70 script-and-commit pairs that match and 14 where the script was not in the folder; we did not copy those versions.

## Files here

49 code and configuration files and 6 result files, copied byte for byte from `experiments/2026-10-14_q3_q35_simhonest_purchase/` at commit `c7fcab33370c95e0399944716d4f0b0f8d5d1cba` of our internal repository. 90 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
