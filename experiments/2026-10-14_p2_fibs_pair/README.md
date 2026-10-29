# 2026-10-14_p2_fibs_pair

2026-10-14 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Does FAR.AI's published FIBS deception probe, applied unchanged, catch Qwen3.5-27B's simulated attacks as often as its real ones?

## What we did

Scored the converted FIBS linear probes (mean of sigmoid outputs over layers 19, 27, 35, 43, 51 and 58, as in their scorer) on the corrected-render runs of the 196-task stratum (real runs 1 and 2, simulated, the two real honest arms and the third honest batch) at reply, command and end-of-call readouts, with a planted gate and a descriptive pair comparison; its scorer and record became the reproduction anchor for later scale-up scoring.

## What we found

Reply readout: attack-vs-honest gate AUROC 0.766; real catch 9.7% at 1%, 32.3% at 5%; paired sim − run 1 at 1% +0.5 points (95% −4.3 to +5.5) on 183 tasks (fibs_scores_2026-10-14_124948.printout.txt).

## In the post?

In the post: Table 1 and Appendix A (deception probe), via the scoring path

## Status

Current (pilot scores later re-scored at full length for the final pooled analysis)

## Builds on

[2026-10-14_q2_fibs_probe](../2026-10-14_q2_fibs_probe/) (weights)

## Main scripts

- [`run_fibs_score.py`](run_fibs_score.py)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_fibs_pair/` at commit `d2d1e0e76810656b3a5367825e3ec8a3b9eede43` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
