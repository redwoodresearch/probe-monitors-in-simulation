# 2026-10-14_p2_own_sim_trained

2026-10-14 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

(Build side of the training-domain test) Fit probes on Qwen3.5-27B's own simulated runs, seal their scores, and build a label-permutation reference for whether they separate their classes.

## What we did

Fitted two simulation-trained probes (simulated attack vs simulated honest, and simulated attack vs real honest; five folds, tool-output and command readouts at layers 16, 32 and 48) on H100s with certified convergence, wrote open and sealed score files for the independent analysis ([2026-10-14_own_s_09](../2026-10-14_own_s_09/)), ran a timing pilot for the permutation reference and then 200 label permutations within main tasks per cell.

## What we found

Fit record: 60 of 60 fits certified, positive control identical on 30 of 30 folds (commit d8264df10 message); label-permutation separability check: reply layers 16/32/48 and command layer 32 separability established at Monte Carlo p-value 1/201, all 6 cells complete after a resume (commit messages 2cac63df3, ae4b07486).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_q3_q35_simhonest_purchase](../2026-10-14_q3_q35_simhonest_purchase/) (simulated honest runs); [2026-10-14_priority1_spec_09](../2026-10-14_priority1_spec_09/) (its appended training-domain test)

## Main scripts

- [`run_owns.py`](run_owns.py)

Result files the index names:

- [`prep_owns_2026-10-14_231856.json`](results/prep_owns_2026-10-14_231856.json)
- `fit_2026-10-14_231946.json` (not copied: result left out to keep the release under 120 MB)
- `pack_sim_sealed_2026-10-14_233605.json` (not copied: result left out to keep the release under 120 MB)
- [`g2pilot_h100_2026-10-14_185203.json`](results/g2pilot_h100_2026-10-14_185203.json)
- [`g2_2026-10-14_232441.json`](results/g2_2026-10-14_232441.json)
- [`g2_2026-10-15_001212.json`](results/g2_2026-10-15_001212.json)
- `g2_release_2026-10-15_000520.json` (not copied: strings over 300 characters)
- `g2_release_2026-10-15_002349.json` (not copied: strings over 300 characters)

## Files here

1 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-14_p2_own_sim_trained/` at commit `ae4b07486fedf6ac89adb636e1d98c8d09ef3630` of our internal repository. 15 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
