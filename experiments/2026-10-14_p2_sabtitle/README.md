# 2026-10-14_p2_sabtitle

2026-10-14 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Do the earlier sabotage-command probes and the cross-reader main probes, read as probe pairs under the frozen analysis plan for the main on-policy question, show a simulated-vs-real catch gap on Qwen3.5-27B's runs?

## What we did

Scored frozen weights: three sabotage probes (the first real-trained sabotage gate, a simulation-trained probe, and a later refit from the sabotage scale-up) at layer-30 end-of-call rows (call and run level, 128 tasks), and the main probes trained on GPT-5.6-Sol's runs (Qwen3-32B and Seed-OSS-36B readers, layer 48, cue-carrying runs excluded, 183 tasks), with thresholds from the second and third real honest batches, paired bootstraps, a label-shuffle null and supplements.

## What we found

The original real-trained sabotage gate, per call at 5%: sim − real −3.8 points (95% −12.7 to +5.0). Previously selected main probe configurations (Qwen3-32B reader, layer 48, honest-B thresholds) at 5%: recipe (b), simulated-attack versus simulated-honest training, +9.3 (95% +3.8 to +14.8), mixed recipe (c), simulated-attack versus real-honest training, +17.9 (95% +12.3 to +23.6) (sabotage_spec and titlecells_spec printouts).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_q3_onpolicy_titlecells](../2026-10-14_q3_onpolicy_titlecells/); [2026-10-14_priority1_spec_09](../2026-10-14_priority1_spec_09/)

## Main scripts

- [`run_sabotage_scores.py`](run_sabotage_scores.py)
- [`run_titlecells_spec.py`](run_titlecells_spec.py)
- [`run_null.py`](run_null.py)
- [`analyze_sabotage_spec.py`](analyze_sabotage_spec.py)
- [`analyze_supplement.py`](analyze_supplement.py)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_p2_sabtitle/` at commit `72fc247729086f0d7ab3f710f8fd0a0d1930ce9e` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
