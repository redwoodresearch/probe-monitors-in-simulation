# 2026-10-13_step2_sim_rewrite

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Step 2, simulator content alone: with the agent's real commands held fixed, does rewriting every tool reply with the simulator reproduce the mixed probe's overstatement?

## What we did

For 30 real GPT-5.6-Sol honest runs (seed 20261013), the September GPT-6-Astra simulator rewrote each tool reply given the real history (API cap \$560), the rewritten transcripts were assembled with byte-identity checks, staged and extracted with Qwen3-32B layers 48 and 63; scoring ran in [2026-10-13_why_steps](../2026-10-13_why_steps/).

## What we found

identical_breakdown_2026-10-13_180314.json: of the rewritten replies, 536 changed and 319 stayed byte-identical to the real reply (bash 283 changed vs 230 identical; str_replace 179 vs 40). Scores and the Step 2 reading are in [2026-10-13_why_steps](../2026-10-13_why_steps/) (why_step2_2026-10-13_230153.json).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

September simulator (Oct 7 simulate code); overlap800 references

## Main scripts

- [`build_rewrite_records.py`](build_rewrite_records.py)
- [`run_rewrite_simulate.py`](run_rewrite_simulate.py)
- [`assemble_rewritten.py`](assemble_rewritten.py)
- [`stage_simrewrite.py`](stage_simrewrite.py)
- [`run_extract_qwen3_simrewrite.py`](run_extract_qwen3_simrewrite.py)
- [`analyze_reply_map.py`](analyze_reply_map.py)

Result files the index names:

- [`identical_breakdown_2026-10-13_180314.json`](results/identical_breakdown_2026-10-13_180314.json)
- [`extract_2026-10-13_q2_step2_simrewrite_qwen3-32b_l48-63_v1_2026-10-13_180101.json`](results/extract_2026-10-13_q2_step2_simrewrite_qwen3-32b_l48-63_v1_2026-10-13_180101.json)
- `s3_manifest_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

8 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_step2_sim_rewrite/` at commit `c87d64e5ccd44c4ea458b0b168014abc0c3514bb` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
