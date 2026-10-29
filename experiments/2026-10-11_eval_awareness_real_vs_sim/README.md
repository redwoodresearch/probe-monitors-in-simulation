# 2026-10-11_eval_awareness_real_vs_sim

2026-10-11 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Does an evaluation-awareness probe read real and formatting-fixed simulated runs of the same task differently, by more than it reads two real runs differently?

## What we did

Scored the fixed-simulator test's 939 formatting-fixed simulated, first real and second real GPT-5.6-Sol attack runs with frozen Qwen3.5-27B evaluation-awareness probes (main readout at the token that starts each assistant turn, layer 8, with content-token readouts beside it) on Modal, after confirming the opening turns are byte-identical across arms; a planted evaluation line far from the readout served as a sensitivity check.

## What we found

README: the planted line moves the layer-8 assistant-turn-start readout +1.64 logits in 900 of 900 cells; ratio of simulated-to-real over real-to-real distance R = 1.014 [0.942, 1.091], 'no more than real-vs-real'; \$16.07 billed. Closed as too few (the cue registers weakly, 1-12% of its synthetic effect), needs a new design.

## In the post?

Not in the post.

## Status

Exploratory (closed as too few)

## Builds on

[2026-10-10_eval_awareness_probe](../2026-10-10_eval_awareness_probe/) (frozen probes); [2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (fixed-simulator test inputs)

## Main scripts

- [`run_hash_turns.py`](run_hash_turns.py)
- [`run_score_claimb.py`](run_score_claimb.py)
- [`analyze_real_vs_sim.py`](analyze_real_vs_sim.py)
- [`analyze_planted_power.py`](analyze_planted_power.py)
- [`analyze_pooled_reference.py`](analyze_pooled_reference.py)
- [`plot_real_vs_sim.py`](plot_real_vs_sim.py)

Result files the index names:

- `real_vs_sim_analysis_2026-10-11_012129.json` (not copied: result left out to keep the release under 120 MB)
- `real_vs_sim_figure_2026-10-11_014705.png` (not copied: result file type not copied (.png))
- [`claimb_scores_pilot_2026-10-11_004650.json`](results/claimb_scores_pilot_2026-10-11_004650.json)
- [`claimb_scores_rest_2026-10-11_005757.json`](results/claimb_scores_rest_2026-10-11_005757.json)
- [`hash_turns_2026-10-11_004048.json`](results/hash_turns_2026-10-11_004048.json)

## Files here

6 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_eval_awareness_real_vs_sim/` at commit `d2b569b5450d78a3804468548f469fbffb9950ae` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
