# 2026-10-15_q3_sp493

2026-10-15 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Run the paper's other 493 scale-up tasks so the comparison covers (almost) every task in the simulated control evaluations paper.

## What we did

Two real-only launches (first and second real attack per cell, plus honest runs on 37 new main tasks) on the frozen harness of the +311 added tasks; preflight, run lists, approach checks and per-arm status records; the simulated arms were separate follow-ups.

## What we found

493 cells over 291 mains; 1,060 real runs planned (README).

## In the post?

In the post: Figure 1 and all task-level results (part of the 1,313 tasks)

## Status

Current

## Builds on

[2026-10-15_q3_interleaved_run3_sim2](../2026-10-15_q3_interleaved_run3_sim2/) (simulated arm)

## Main scripts

- [`make_sp493_tasks.py`](make_sp493_tasks.py)
- [`stage_sp493.sh`](stage_sp493.sh)
- [`launch_sp493.sh`](launch_sp493.sh)
- [`start_sp493.sh`](start_sp493.sh)
- [`gate_sp493.py`](gate_sp493.py)
- [`verify_v2_sp493.py`](verify_v2_sp493.py)

Result files the index names:

- `sp493_facts_2026-10-15_165730.json` (not in this folder at the source commit)
- `c17_sp493_ROUTEC_2026-10-18_100702.json` (not copied: result file 1 MB or larger)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_q3_sp493/` at commit `0b973ed8815f58e0367495a2f835ae8b42361f29` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 6 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
