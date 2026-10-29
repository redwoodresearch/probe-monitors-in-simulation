# 2026-10-14_p2_apollo

2026-10-14 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does Apollo Research's deception probe, retrained on Qwen3.5-27B (frozen on 10-09, layer 31), catch simulated attacks as often as real ones on Qwen3.5's own runs?

## What we did

Scored the frozen Apollo-recipe probe (per-token logistic regression, layer 31 chosen on Apollo's own development data) on the corrected-render pilot runs (196 tasks; real runs 1 and 2, simulated, the two real honest arms and the third honest batch) at reply, command and end-of-call readouts; a convergence certificate failed, so a tight refit (Amendment 2) was scored instead.

## What we found

Tight refit, reply readout: attack-vs-honest gate AUROC 0.750; real catch 2.1% at 1% and 20.2% at 5%; paired sim − run 1 at 1% +1.6 points (95% −0.6 to +4.1) on 183 tasks (scores_2026-10-14_145858.printout.txt). Kept as a descriptive row outside the probe family (verdict under the experiment's written rules); the post uses the FIBS deception probe instead.

## In the post?

Not in the post.

## Status

Exploratory (dropped from the family)

## Builds on

[2026-10-09_q2_deception_apollo](../2026-10-09_q2_deception_apollo/) (probe weights); [2026-10-14_q35_sol_avg_POSTHOC](../2026-10-14_q35_sol_avg_POSTHOC/) (corrected-B manifest)

## Main scripts

- [`run_fold_apollo.py`](run_fold_apollo.py)
- [`run_convergence.py`](run_convergence.py)
- [`run_fix.py`](run_fix.py)

Result files the index names:

- `fold_2026-10-14_140044.json` (not copied: result left out to keep the release under 120 MB)
- `convergence_2026-10-14_143716.json` (not copied: result left out to keep the release under 120 MB)
- `fix_2026-10-14_145508.json` (not copied: result left out to keep the release under 120 MB)
- `fit_tight_2026-10-14_145508.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_apollo/` at commit `d77a59d0a7a73f44f863b41d8f765c47cf2f30df` of our internal repository. 17 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
