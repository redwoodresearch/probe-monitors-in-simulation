# 2026-10-13_claimB_offered_tools

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Simulated runs were offered three extra tools (grep, list_dir, read_file); does any activation reader render the offered-tool list, which would put a non-reply simulated-vs-real difference into every readout?

## What we did

Code reading of each reader pipeline's chat-template call (Qwen3-32B, Qwen3.5-27B, Olmo-3.1-32B, Llama-3.3-70B, Seed-OSS-36B) and of the staged rows; Parts 2 and 3 (per-cohort tool-set counts) were planned on Modal CPU. \$0.

## What we found

render_calls_CODEREAD_2026-10-13_172630.md: no reader passes tools= to the chat template and staged rows have no tools field, so the offered-tool list never reaches any reader; the three extra tools were offered to the simulator model (GPT-6-Astra), not to the agent.

## In the post?

Not in the post.

## Status

Exploratory (Part 1 only; Parts 2 and 3 have no committed result)

## Builds on

[2026-10-13_claimB_read10](../2026-10-13_claimB_read10/) (raised the question)

## Main scripts

- none (code reading
- RULE.md only)

Result files the index names:

- [`render_calls_CODEREAD_2026-10-13_172630.md`](results/render_calls_CODEREAD_2026-10-13_172630.md)

## Files here

0 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_claimB_offered_tools/` at commit `0a301d3ad4153119876e0521b10433909826d38e` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
