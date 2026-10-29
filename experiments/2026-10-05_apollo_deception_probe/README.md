# 2026-10-05_apollo_deception_probe

2026-10-05 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does Apollo Research deception probe, retrained on Qwen3-32B with their data and recipe, detect deception on their Roleplaying evaluation set?

## What we did

Ported Apollo deception-detection code (src/probe_shift/apollo.py), trained a per-token logistic regression on their instructed honest/deceptive statement pairs at Qwen3-32B layers 16, 24, 32, 40 and 48, and scored their Roleplaying set (203 deceptive vs 84 honest).

## What we found

README table: Roleplaying area under the ROC curve (AUROC) 0.588 to 0.818 across five layers, below Apollo 0.9637 on Llama-3.3-70B; held-out instructed pairs 1.000 (README).

## In the post?

Not in the post.

## Status

Superseded by [2026-10-06_apollo_qwen_gap](../2026-10-06_apollo_qwen_gap/) (every-layer sweep reproduces these five numbers and finds a better layer)

## Builds on

Nothing earlier in this index.

## Main scripts

- [`run_apollo_probe.py`](run_apollo_probe.py)

Result files the index names:

- `apollo_probe_2026-10-05_230456.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-05_apollo_deception_probe/` at commit `c9d1266001f17581b0fb69feabf38c98a74a632b` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
