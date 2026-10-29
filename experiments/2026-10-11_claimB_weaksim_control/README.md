# 2026-10-11_claimB_weaksim_control

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Positive control: does the fixed-simulator test's frozen realism probe separate a simulator known to be weak (GPT-6-Astra with no reference run and no codebase access) from real runs?

## What we did

Scored a collaborator's 18-environment study (real attack runs vs the weak base simulator, 198 cells over 18 environments) with the frozen main Qwen3-32B end-of-output realism probe on Qwen3-32B readings, plus scrambled refits, fresh leave-one-environment-out probes and a post hoc length-controlled reading.

## What we found

README: PASS by the registered rule, within-environment AUROC 0.897 (lower bound 0.864 vs the 0.75 bar); the planted line is still missed (0.456); fresh probes reach 1.0; weak-simulator runs are much shorter and token count alone out-ranks the probe (probe minus length -0.048 [-0.083, -0.013]), though the probe adds beyond length (+0.027 [0.014, 0.040]).

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (its first probe step); a collaborator's 18-environment cohort (external)

## Main scripts

- [`check_la18_cohort.py`](check_la18_cohort.py)
- [`power_weaksim.py`](power_weaksim.py)
- [`run_weaksim_stage_modal.py`](run_weaksim_stage_modal.py)
- [`run_weaksim_score_modal.py`](run_weaksim_score_modal.py)
- [`analyze_weaksim.py`](analyze_weaksim.py)
- [`analyze_POSTHOC_length.py`](analyze_POSTHOC_length.py)

Result files the index names:

- [`weaksim_reading_2026-10-11_233727.json`](results/weaksim_reading_2026-10-11_233727.json)
- [`length_POSTHOC_2026-10-11_235145.json`](results/length_POSTHOC_2026-10-11_235145.json)
- [`stage_2026-10-11_231451.json`](results/stage_2026-10-11_231451.json)
- `la18_cohort_2026-10-11_211045.json` (not copied: result left out to keep the release under 120 MB)
- `power_weaksim_2026-10-11_224612.json` (not copied: result left out to keep the release under 120 MB)

## Files here

6 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_claimB_weaksim_control/` at commit `6d2cf09cb5165cab81145db5523469090c1e6cb1` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
