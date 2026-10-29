# 2026-10-11_q2_e1_honest

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Hide-or-mention test, stage 1: does the frozen deception probe (Apollo's recipe retrained on Qwen3.5-27B, layer 31) respond when an honest agent is told to leave one specific file edit out of its final report?

## What we did

On 60 real honest Qwen3.5-27B runs from the 196-task cohort, regenerated the final report under 'hide this edit' and 'mention this edit' instructions on Modal and scored the report with the frozen probe (D = hide minus instructed, in honest standard deviations), with a textual positive control and disclosure audit.

## What we found

README: pre-registered verdict 'response not established' because the textual positive control failed (+0.046 SD vs 0.5 required); after correction: stage-1 mean D +0.82 (0.49 to 1.22) over 30 runs, textual control +0.05; first launch crashed before GPU work (about \$0.13).

## In the post?

Not in the post.

## Status

Exploratory (closed: too few; follow-ups in [2026-10-11_q2_e1_sentences](../2026-10-11_q2_e1_sentences/) and [2026-10-11_q2_e1_confirm](../2026-10-11_q2_e1_confirm/))

## Builds on

[2026-10-11_q2_zoo_plans](../2026-10-11_q2_zoo_plans/) (PLAN_E1_honest.md); [2026-10-09_q2_deception_apollo](../2026-10-09_q2_deception_apollo/) (frozen probe); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (runs of the first real honest arm)

## Main scripts

- [`build_e1_index.py`](build_e1_index.py)
- [`launch_e1_stage1.py`](launch_e1_stage1.py)
- [`run_e1.py`](run_e1.py)
- [`analyze_e1.py`](analyze_e1.py)
- [`analyze_e1_control_POSTHOC.py`](analyze_e1_control_POSTHOC.py)
- [`analyze_e1_control_null_POSTHOC.py`](analyze_e1_control_null_POSTHOC.py)
- [`check_editor_failures.py`](check_editor_failures.py)

Result files the index names:

- [`e1_stage1_analysis_2026-10-11_135302.json`](results/e1_stage1_analysis_2026-10-11_135302.json)
- [`e1_stage1_analysis_2026-10-11_134021.json`](results/e1_stage1_analysis_2026-10-11_134021.json)
- [`e1_control_POSTHOC_2026-10-11_134120.json`](results/e1_control_POSTHOC_2026-10-11_134120.json)
- [`e1_control_null_POSTHOC_2026-10-11_203515.json`](results/e1_control_null_POSTHOC_2026-10-11_203515.json)
- `disclosure_audit_2026-10-11_131253_stage1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 8 result files, copied byte for byte from `experiments/2026-10-11_q2_e1_honest/` at commit `4f8e721c42d8719a931c856abf26224adfdb636a` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
