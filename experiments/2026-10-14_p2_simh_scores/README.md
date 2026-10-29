# 2026-10-14_p2_simh_scores

2026-10-14 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Score Qwen3.5-27B's simulated honest runs (and later every group of runs at full length, without the 32,768-token cut) with the committed scoring code of the four published probes: the deception probe trained on the FIBS dataset, the refusal direction, the harmfulness direction and the high-stakes probe.

## What we did

Started as the scoring of the simulated honest pilot for the fully simulated comparison (simulated attacks thresholded on simulated honest runs); on other branches the folder holds the full-length scoring launchers, the launchers for the second batch of real honest runs and for simulated honest runs, and checks for runs missing from activation extraction. The latest branch keeps only a 2026-10-17 regression test showing that the new scoring code reproduces the old code and the stored production score files.

## What we found

a48_regress_2026-10-17_133427.json: old vs new scorer bitwise and new vs stored production files, verdict PASS. Pilot and later scoring records are on other branches (e.g. commit 448813b9b).

## In the post?

In the post: Tables 1, 5 and 8 indirectly (scores of the four published probes)

## Status

Current

## Builds on

[2026-10-14_p2_fibs_pair](../2026-10-14_p2_fibs_pair/); [2026-10-14_p2_refusal_harm](../2026-10-14_p2_refusal_harm/); [2026-10-14_p2_highstakes](../2026-10-14_p2_highstakes/); [2026-10-14_q3_q35_simhonest_purchase](../2026-10-14_q3_q35_simhonest_purchase/)

## Main scripts

- `run_simh_score.py` (not in this folder at the source commit)
- [`launchers/cutfull/launch_cutfull.py`](launchers/cutfull/launch_cutfull.py)
- [`launchers/cutfull/regress_oldnew.py`](launchers/cutfull/regress_oldnew.py) (not on the latest ref)

Result files the index names:

- `a48_regress_2026-10-17_133427.json` (not copied: strings over 300 characters)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 1 result file the index names, they match in 2 script-and-commit pairs.

## Files here

0 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_simh_scores/` at commit `0b973ed8815f58e0367495a2f835ae8b42361f29` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). We also copied 2 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: low). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
