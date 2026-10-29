# 2026-10-11_q2_n2_honest

2026-10-11 · part of [Side question: does the agent behave differently in simulation?](../../docs/experiments.md#side-question-does-the-agent-behave-differently-in-simulation)

## Question

Does GPT-5.6-Sol behave differently in simulated honest runs than in real ones on two scripted behaviours (touching log, audit or shell-history paths; reading files back after writing), against the difference between two real runs?

## What we did

\$0 laptop counts on the 163 same-batch tasks: per-run shares of each behaviour, real minus simulated with real run 1 minus run 2 as yardstick, equivalence margins and planted checks; post hoc batch and extractor checks.

## What we found

README: log and audit paths agree within plus or minus 1 point (real minus simulated -0.01 [-0.39, +0.38]; two real runs +0.12); read-backs too insensitive to read (a margin-sized plant detected 0 of 200).

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_q2_zoo_plans](../2026-10-11_q2_zoo_plans/) (PLAN_N2.md)

## Main scripts

- [`analyze_n2_honest.py`](analyze_n2_honest.py)
- [`plot_n2_honest.py`](plot_n2_honest.py)
- [`analyze_n2_batch_POSTHOC.py`](analyze_n2_batch_POSTHOC.py)
- [`analyze_n2_extractor_impact_POSTHOC.py`](analyze_n2_extractor_impact_POSTHOC.py)
- [`analyze_n2_extractor_v2_POSTHOC.py`](analyze_n2_extractor_v2_POSTHOC.py)

Result files the index names:

- [`n2_honest_2026-10-11_112022.json`](results/n2_honest_2026-10-11_112022.json)
- `n2_honest_2026-10-11_112022.png` (not copied: figure (not copied))
- [`n2_batch_POSTHOC_2026-10-11_113021.json`](results/n2_batch_POSTHOC_2026-10-11_113021.json)
- [`n2_extractor_impact_POSTHOC_2026-10-11_114530.json`](results/n2_extractor_impact_POSTHOC_2026-10-11_114530.json)
- [`n2_honest_2026-10-11_114424_extractor-v2_POSTHOC.json`](results/n2_honest_2026-10-11_114424_extractor-v2_POSTHOC.json)

## Files here

5 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-11_q2_n2_honest/` at commit `0f17af675de48720268ec742aa0a5e1c590e56dd` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
