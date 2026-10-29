# 2026-10-14_convergence_audit_09

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Did the probe fitter stop too early (its gradient tolerance is not scaled by the regularization strength C), and does a tightly converged refit change the main probes' catch rates or the choices of an earlier recipe that picks layer, readout and C from simulated runs only?

## What we did

Refit the earlier phase's main probes (Qwen3-32B and Seed-OSS-36B reading GPT-5.6-Sol's runs with all formatting repairs, token-averaged readout, layer 48, trained on simulated attack vs simulated honest or on real attack vs real honest runs, 20 nested models each) and that recipe's C = 0.001 candidates with a certified tight tolerance, then compared stored and tight scores, catches and objective values.

## What we found

README: Part 1 (C 0.01) the numbers stand, largest catch move 0.58 points at 5% and 0.45 at 1%, as-run fits 0.27–0.37% above the tight objective; Part 2 (C 0.001): all registered choices stayed unchanged except Seed-OSS-36B fold 2, which switched from layer 63, C=0.01 to layer 40, C=0.001, both at the last-token readout.

## In the post?

Not in the post.

## Status

Current

## Builds on

2026-10-13 title-cell fits (why-steps job)

## Main scripts

- [`run_conv_titlecells.py`](run_conv_titlecells.py)
- [`run_conv_recipe1.py`](run_conv_recipe1.py)
- [`recipe_select_757d07ac.py`](recipe_select_757d07ac.py)
- [`recipe_select_bf121ce2.py`](recipe_select_bf121ce2.py)

Result files the index names:

- [`conv_titlecells_2026-10-14_062715.json`](results/conv_titlecells_2026-10-14_062715.json)
- [`conv_recipe1_2026-10-14_064421.json`](results/conv_recipe1_2026-10-14_064421.json)

## Files here

4 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-14_convergence_audit_09/` at commit `97121674eae74153a9ff6c1fc4cc178b8f9ccc7e` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
