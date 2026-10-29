# 2026-10-13_q2_llama_leftover

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Do the other simulator fixes already frozen by the project (insert form, dpkg, clock, trailing newlines) account for Llama-3.3-70B's leftover gap after the main formatting repairs to all 939 cells?

## What we did

Built 'repaired-plus' simulated runs (865 changed) on top of the repairs to all 939 cells, re-extracted Llama-3.3-70B activations (2 H100s per shard) and rescored with its frozen probes on the 939 cells, with a prefix gate. \$48.84 billed.

## What we found

README: paired shift repaired-plus minus repaired +0.29 [-0.36, +1.19] at 5% and +0.05 at 1% ('no change'); real second run minus repaired-plus stays +5.51 [+2.55, +8.06] at 5% and +6.28 at 1%; predicted shift +2.3 missed.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_q2_llama_reader](../2026-10-12_q2_llama_reader/) (part 2, frozen probes); the repaired arm of all 939 cells

## Main scripts

- [`run_extract_llama_plus.patch`](run_extract_llama_plus.patch)
- [`run_score_plus_llama.py`](run_score_plus_llama.py)
- [`analyze_plus_llama.py`](analyze_plus_llama.py)
- [`count_local_sites.py`](count_local_sites.py)

Result files the index names:

- [`plus_scores_llama_2026-10-13_024006.json`](results/plus_scores_llama_2026-10-13_024006.json)
- [`local_sites_2026-10-13_012231.json`](results/local_sites_2026-10-13_012231.json)
- `plus_reading_llama_2026-10-13_024135.json` (not in this folder at the source commit)
- `_plus_2026-10-13_023705.json` (not in this folder at the source commit)

## Files here

4 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-13_q2_llama_leftover/` at commit `80fde6fd08fe12a86d703c4348c933f23e1d411a` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
