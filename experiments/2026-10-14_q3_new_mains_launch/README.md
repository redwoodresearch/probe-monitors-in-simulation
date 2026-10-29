# 2026-10-14_q3_new_mains_launch

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Run the 360 new-main-task cells (outside the paper's 542 mains) to add main tasks, which is what tightens the task-resampled intervals.

## What we did

Launched on devboxes with the scale-up's frozen harness plus the new_mains_v1 construction mode: per cell Qwen3.5-27B's two real honest arms, two real attacks in one launch, and a September simulated attack in per-split follow-ups (five splits), with reruns of the fourth and fifth splits (the fifth a rate-limit rerun), error censuses and an epoch-2 quote; all-in cap \$11,000.

## What we found

Runs launched and censused split by split (112 simulator error-census files, final-state and supersession records in results); no single summary file; feeds the 1,360 launched tasks.

## In the post?

In the post: Appendix A runs (the tasks with new main tasks, part of the 1,313 tasks); Appendix I

## Status

Current

## Builds on

[2026-10-14_q3_new_mains_mode](../2026-10-14_q3_new_mains_mode/); [2026-10-14_q3_scaleup_311](../2026-10-14_q3_scaleup_311/) (harness); [2026-10-14_q2_scaleup_cohort](../2026-10-14_q2_scaleup_cohort/)

## Main scripts

- [`box_setup_nm.sh`](box_setup_nm.sh)
- checks/census_\*.sh
- [`checks/attempt/attempt_rate_311.py`](checks/attempt/attempt_rate_311.py)
- [`checks/compose_race_check.py`](checks/compose_race_check.py)

Result files the index names:

- [`s4_final_states.json`](results/s4_final_states.json)
- [`s4r_selection.json`](results/s4r_selection.json)
- `s5r_supersession_map.json` (not copied: result left out to keep the release under 120 MB)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

143 code and configuration files and 14 result files, copied byte for byte from `experiments/2026-10-14_q3_new_mains_launch/` at commit `3728476982b9682798446ff982ef20bdad3aae8c` of our internal repository. 186 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
