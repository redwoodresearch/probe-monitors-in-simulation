# 2026-10-14_q2_scaleup_cohort

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Which extra LinuxArena tasks should scale Qwen3.5-27B's on-policy cohort from 196 tasks toward the paper's 1,000 and beyond?

## What we did

Drew task lists (files only, nothing launched): +311 added cells on new main tasks (total 507), the paper's remaining 804 cells (total 1,000), a count of 360 new main tasks outside the paper's 542, and an eligibility check of the remaining 493 paper cells.

## What we found

READMEs: list of the +311 added tasks (tasks 197–507) and list of the paper's +804 remaining cells drawn; 360 new main tasks can each take a side task (317 already ran honest) but were blocked at build time; all 493 remaining paper cells eligible. The launched cohort is 196 + 311 + 360 + 493 = 1,360 tasks, matching the post's Appendix I.

## In the post?

In the post: Appendix A and Appendix I (task lists behind the 1,360 launched tasks)

## Status

Current

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (196-task file)

## Main scripts

- [`select_cohort.py`](select_cohort.py)
- [`count_new_mains.py`](count_new_mains.py)
- [`count_remaining.py`](count_remaining.py)

Result files the index names:

- `scaleup_tasks_plus311_2026-10-14_125135.json` (not copied: result left out to keep the release under 120 MB)
- `scaleup_tasks_plus804_2026-10-14_125135.json` (not copied: result left out to keep the release under 120 MB)
- [`scaleup_cohort_record_2026-10-14_125135.json`](results/scaleup_cohort_record_2026-10-14_125135.json)
- `new_mains_counts_2026-10-14_163152.json` (not copied: result left out to keep the release under 120 MB)
- `remaining490_counts_2026-10-14_160219.json` (not copied: result left out to keep the release under 120 MB)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 5 result files the index names, they match in 6 script-and-commit pairs, and in 3 the script was not in the folder at that commit.

## Files here

3 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_q2_scaleup_cohort/` at commit `8a3af7b1ec18772fad34217ade30150fcbb93a82` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
