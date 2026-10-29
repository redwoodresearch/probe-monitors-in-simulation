# 2026-10-06_apollo_llama_check

2026-10-06 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does our port of Apollo deception probe reproduce Apollo own probe on the model Apollo used (Llama-3.3-70B)?

## What we did

Ran Apollo code unchanged and our port side by side on Llama-3.3-70B-Instruct at layer 22 (2 H100s), comparing Roleplaying AUROC and per-dialogue scores against Apollo shipped scores.

## What we found

README table: Apollo shipped 0.9637, Apollo rerun 0.9635, port retrained 0.9631; per-dialogue Pearson r 0.9999997 between port and shipped probe (README).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-05_apollo_deception_probe](../2026-10-05_apollo_deception_probe/)

## Main scripts

- [`run_apollo_llama.py`](run_apollo_llama.py)

Result files the index names:

- `apollo_original_2026-10-06_112113.json` (not copied: strings over 300 characters)
- [`port_llama_2026-10-06_105756.json`](results/port_llama_2026-10-06_105756.json)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-06_apollo_llama_check/` at commit `8029fcfe76a39bf5a30f8ac52220903a922826c0` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
