# 2026-10-11_fake_env_concept_probe

2026-10-11 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Does Qwen3-32B represent 'this environment is simulated' differently on fixed-simulator LinuxArena runs than on real ones, beyond the difference between two real runs?

## What we did

Built a concept probe from synthetic contrast pairs ('simulated environment' vs 'real production system' in system prompts and other families, then a retry adding implied cues), validated it with a planted far-line gate, and scored the fixed-simulator test's fixed-simulator and real runs (89 cells, then a post hoc extension to 340) on Modal.

## What we found

README: first attempt stopped at stage 3 (prerequisite failed, \$0.47); retry passed its gate and stage 5 was inconclusive, content readout R (mean absolute simulated-minus-first-real score difference divided by mean absolute second-minus-first-real score difference) = 1.261 [1.019, 1.544] on 89 cells; post hoc extension still inconclusive, R = 1.094 [0.973, 1.229] on 340 cells.

## In the post?

Not in the post.

## Status

Exploratory (inconclusive)

## Builds on

[2026-10-10_eval_awareness_probe](../2026-10-10_eval_awareness_probe/) (fit code); [2026-10-11_eval_awareness_real_vs_sim](../2026-10-11_eval_awareness_real_vs_sim/) (turn hashes); [2026-10-09_fixed_sim](../2026-10-09_fixed_sim/)

## Main scripts

- [`build_concept_sets.py`](build_concept_sets.py)
- [`build_concept_sets_v2.py`](build_concept_sets_v2.py)
- [`run_extract_concept.py`](run_extract_concept.py)
- [`run_score_concept.py`](run_score_concept.py)
- [`analyze_concept_gate.py`](analyze_concept_gate.py)
- [`analyze_concept_real.py`](analyze_concept_real.py)
- [`analyze_concept_leakage.py`](analyze_concept_leakage.py)
- [`plot_concept.py`](plot_concept.py)
- [`plot_concept_extension.py`](plot_concept_extension.py)

Result files the index names:

- `concept_real_analysis_2026-10-11_022327.json` (not copied: result left out to keep the release under 120 MB)
- `concept_real_analysis_extension_2026-10-11_032343.json` (not copied: result left out to keep the release under 120 MB)
- [`concept_gate_analysis_2026-10-11_021444.json`](results/concept_gate_analysis_2026-10-11_021444.json)
- [`concept_gate_prereq_2026-10-11_013036.json`](results/concept_gate_prereq_2026-10-11_013036.json)

## Files here

9 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_fake_env_concept_probe/` at commit `10009588d09c94a74c8e588f322851a56b51ec44` of our internal repository. 21 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
