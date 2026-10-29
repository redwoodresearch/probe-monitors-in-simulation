# 2026-10-16_behaviour_full

2026-10-16 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Did the agent behave differently in simulation: steps, tool calls, output tokens and how often it attempted the side task?

## What we did

Full-cohort behaviour comparison (simulated vs real run 1 and run 2) per task set, run length vs attempt near the 200-step cap, and main-task-clustered bootstrap intervals for the post on all launched cells and the 1,313-task cohort; Modal CPU.

## What we found

1,313 tasks: clear attempt 95.7% sim vs 97.0% run 1; non-attempt runs' median 192 steps in simulation vs 67 for attempted runs (main_bootstrap_2026-10-16_144720 printout).

## In the post?

In the post: Appendix G; footnote on attempts

## Status

Current

## Builds on

[2026-10-15_attempt_rate](../2026-10-15_attempt_rate/); [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) main record

## Main scripts

- [`run_behaviour_full.py`](run_behaviour_full.py)
- [`run_cap_summary.py`](run_cap_summary.py)
- [`run_main_bootstrap.py`](run_main_bootstrap.py)

Result files the index names:

- `main_bootstrap_2026-10-16_100211.json` (not copied: strings over 300 characters)
- `main_bootstrap_2026-10-16_144720.json` (not copied: strings over 300 characters)
- `behaviour_full_2026-10-16_025753.json` (not copied: result file 1 MB or larger)
- `cap_summary_2026-10-16_033450.json` (not copied: result left out to keep the release under 120 MB)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/4a0acbc17/run_main_bootstrap.py`](as_run/4a0acbc17/run_main_bootstrap.py) from commit `4a0acbc17`: results/main_bootstrap_2026-10-16_100211.json records commit 4a0acbc17, where this script had this version. The later main_bootstrap_2026-10-16_144720 result, which the index cites for the post's Appendix G numbers, records commit c7dd37380, where the script matches the copy in this folder.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 4 result files the index names, they match in 8 script-and-commit pairs, differ in 1, and in 3 the script was not in the folder at that commit. Other result files, from earlier or side runs, give 0 script-and-commit pairs that match, 1 that differ and 2 where the script was not in the folder; we did not copy those versions. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-16_behaviour_full/` at commit `368533e9781853126fc648c45e55066711ea0dec` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 1 file under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
