# 2026-10-09_q2_onpolicy_discrimination

2026-10-09 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Can a classifier tell Qwen3.5-27B's real runs from simulated ones on the 20-task strict rerun (both simulators)?

## What we did

Run-level TF-IDF (whole run and per output) and a Qwen3.5-27B activation probe, folds by main task, 2,000 task resamples and refitting label-swap nulls, with real honest_a vs honest_b as the reference; predictions first.

## What we found

The per-output text classifier separates real from the September simulator (0.65, p = 0.020, borderline) but not today's (0.55); probe medians 0.52-0.61, far below the predictions (0.95 / 0.85) (preliminary).

## In the post?

Not in the post (Appendix E (an early 20-task version of the simulation detector; the post's detector numbers come from later 1,313-task records)).

## Status

Superseded by the 1,313-task simulation detector behind Figure 2 and Appendix E (records in [2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/) and [2026-10-16_honest_sim_detector_all](../2026-10-16_honest_sim_detector_all/))

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (strict rerun); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (extractions, folds)

## Main scripts

- [`run_tfidf_discrimination.py`](run_tfidf_discrimination.py)
- [`run_tfidf_per_output.py`](run_tfidf_per_output.py)
- [`run_probe_discrimination.py`](run_probe_discrimination.py)
- [`analyze_discrimination.py`](analyze_discrimination.py)
- [`check_fork_prefix.py`](check_fork_prefix.py)

Result files the index names:

- `manifest_v1.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_v2.json` (not copied: strings over 300 characters)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_q2_onpolicy_discrimination/` at commit `78b88c632f0a84bae201d386f98f73785b4d8a1a` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
