# 2026-10-14_onpolicy_sim

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

(Plumbing for the patching pilot) Can the September simulator rewrite the first 20 tool outputs of the drawn Qwen3.5-27B runs, and can Qwen3-32B read the original and rewritten versions?

## What we did

A Modal port of the per-call September simulator (GPT-6-Astra through the OpenAI API, key held in a Modal Secret) built simulator records for the patching pilot's drawn runs, simulated the honest and then the attack runs under a 450-call and \$130 cap, tallied the spend ledger, and extracted Qwen3-32B activations for 6 versions of up to 20 runs.

## What we found

Attack arm: 198 simulator calls, all ok, \$43.95 charged, 0 substitutions (sim_tally_attack_2026-10-14_103429.json); honest arm \$39.43 per SPEND.md of the patching folder; activations extracted (extract_summary JSON).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_onpolicy_patching](../2026-10-14_onpolicy_patching/) (draw)

## Main scripts

- [`onpolicy_sim.py`](onpolicy_sim.py)
- [`stage_onpolicy.py`](stage_onpolicy.py)
- [`run_extract_onpolicy_qwen3.py`](run_extract_onpolicy_qwen3.py)
- [`evalzip.py`](evalzip.py)

Result files the index names:

- `sim_tally_honest_2026-10-14_102301.json` (not copied: strings over 300 characters)
- `sim_tally_attack_2026-10-14_103429.json` (not copied: strings over 300 characters)
- [`records_problems_2026-10-14_100242.json`](results/records_problems_2026-10-14_100242.json)
- [`extract_summary_2026-10-14_q2_onpolicy_patching_qwen3-32b_l7_v1.json`](results/extract_summary_2026-10-14_q2_onpolicy_patching_qwen3-32b_l7_v1.json)

## Files here

5 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-14_onpolicy_sim/` at commit `d6016ac8d0a753ed3851c5b34c767c9e2055cd7d` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
