# 2026-10-11_sim_source_audit

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

For every off-policy probe result comparing real and simulated runs, was the simulated arm built from the same real run it is compared against, and do the results hold once that overlap is removed?

## What we did

\$0 read-only audit tabulating each result's simulator source and compared real runs; fix (a) recomputed first-phase results (GPT-5.6-Sol runs read by Qwen3-32B) on cells whose simulator reference was a fresh rollout; fix (b) re-ran the trained attack-detector family (probes separating attack-instructed from honest Qwen3.5-27B runs) on Modal with the two real honest arms' roles swapped (about \$10.75 budget).

## What we found

AUDIT.md: no result compares a simulated attack run with its own source; the overlap is on the honest side (196 of 939 Phase 1 cells; 188 of 191 Phase 2 simulated runs built from the first real honest arm). RESULTS_b.md: the verdicts of the Qwen3.5-27B attack-detector comparison hold under the swap (command end +1.6 to +0.0 [-0.6, +1.1], 10-point gap still ruled out).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/); [2026-10-06_data_checks](../2026-10-06_data_checks/)

## Main scripts

- [`count_sim_sources.py`](count_sim_sources.py)
- [`fixes/rescore_q36.py`](fixes/rescore_q36.py)
- [`fixes/rescore_phase1.py`](fixes/rescore_phase1.py)
- [`fixes/rescore_nano.py`](fixes/rescore_nano.py)
- [`fixes/stamp_arms.py`](fixes/stamp_arms.py)
- [`fixes/resume_b.py`](fixes/resume_b.py)
- [`fixes/compare_b.py`](fixes/compare_b.py)

Result files the index names:

- [`sim_sources_2026-10-11_193731.json`](results/sim_sources_2026-10-11_193731.json)
- `compare_b_2026-10-11_231500.txt` (not copied: lines over 300 characters)

## Files here

9 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_sim_source_audit/` at commit `9869761b285537955b5f1c3cceb5f79478e44389` of our internal repository. 24 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
