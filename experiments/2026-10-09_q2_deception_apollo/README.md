# 2026-10-09_q2_deception_apollo

2026-10-09 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does Apollo's deception probe recipe work on Apollo's other test sets, and does it carry over when retrained on Qwen3.5-27B?

## What we did

Rebuilt Apollo's insider-trading, sandbagging and Alpaca sets, checked port parity on Llama-3.3-70B, then retrained Apollo's probe on Qwen3.5-27B and scored Apollo's test sets (about \$3-5.5).

## What we found

Port parity passed (labels match item for item, scores correlate at r 0.9999999). The Qwen3.5-27B retrain ranks deceptive above honest at 0.875 to 0.9996 across sets, but a threshold set on ordinary chat flags 75% of honest roleplaying answers (preliminary).

## In the post?

Not in the post (the post's deception probe is the published FIBS weights, not this retrain).

## Status

Current (preliminary; the frozen step-2 probe feeds later deception folders)

## Builds on

[2026-10-06_apollo_qwen_gap](../2026-10-06_apollo_qwen_gap/); src/probe_shift/apollo.py

## Main scripts

- [`run_step1_llama_parity.py`](run_step1_llama_parity.py)
- [`run_step2_qwen35.py`](run_step2_qwen35.py)
- [`analyze_apollo_sets.py`](analyze_apollo_sets.py)
- [`analyze_step2_qwen35.py`](analyze_step2_qwen35.py)
- [`analyze_sandbagging_domains.py`](analyze_sandbagging_domains.py)

Result files the index names:

- `manifest_v1.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_v2.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_v3.json` (not copied: result left out to keep the release under 120 MB)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-09_q2_deception_apollo/` at commit `af5e090f667a8c20d5c2ac93aa457e06271a2231` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
