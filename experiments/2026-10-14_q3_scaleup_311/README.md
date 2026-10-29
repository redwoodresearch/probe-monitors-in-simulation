# 2026-10-14_q3_scaleup_311

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Scale the on-policy cohort from 196 to about 507 tasks: run 303 new cells (plus the canary's 8) with two real attack runs in the same launch, one simulated attack and two real honest runs each.

## What we did

Two devboxes plus the canary box ran the frozen September harness in four quarters with Qwen3.5-27B on H200 vLLM servers; simulated attacks ran in follow-up launches; the second quarter's simulated follow-up was rerun for 55 cells; caps up to \$10,000 all-in with on-box gates and negative tests.

## What we found

Run table of 1,550 planned runs over 310 cells: 620 real attacks, 310 first-arm and 310 second-arm real honest runs, 310 simulated attacks (run_table_311.json); unusable-run lists and a reference audit are committed.

## In the post?

In the post: Appendix A runs (the +311 added tasks of the 1,313 tasks)

## Status

Current

## Builds on

[2026-10-14_q3_scaleup_canary](../2026-10-14_q3_scaleup_canary/); [2026-10-14_q2_scaleup_cohort](../2026-10-14_q2_scaleup_cohort/)

## Main scripts

- [`make_tasks_311.py`](make_tasks_311.py)
- [`add_r2.py`](add_r2.py)
- [`check_pairs.py`](check_pairs.py)
- [`compose_race_check.py`](compose_race_check.py)
- [`bootstrap_box.sh`](bootstrap_box.sh)
- [`allin_stop.sh`](allin_stop.sh)

Result files the index names:

- `run_table_311.json` (not copied: result left out to keep the release under 120 MB)
- `unusable_runs_final_main_2026-10-14_1908.jsonl` (not copied: result file type not copied (.jsonl))
- `unusable_runs_final_fu_q134_q2r_2026-10-14_2300.jsonl` (not copied: result file type not copied (.jsonl))
- `reference_audit_summary_2026-10-15.json` (not copied: strings over 300 characters)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

61 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-14_q3_scaleup_311/` at commit `cfdd656c96cd2100c1e2a8d07c05650f2a7754e3` of our internal repository. 65 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
