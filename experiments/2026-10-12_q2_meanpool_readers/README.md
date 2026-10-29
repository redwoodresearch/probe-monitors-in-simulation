# 2026-10-12_q2_meanpool_readers

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Gate row 1: for Olmo-3.1-32B and Llama-3.3-70B, is the reply detector's break about the last-token readout or the layer?

## What we did

Extracted mean-pooled activations for both readers on Modal (Olmo \$59.31; Llama \$297.72 after a restart of the job's coordinating container duplicated shards and a resume) and compared mean-pooled vs last-token detectors at the layers last-token chose (Olmo 32/40/56, Llama 60/70).

## What we found

Olmo mean-pooled gaps -1.30 to -2.48 vs last-token -3.57 to -10.92, verdict "mixed or unclear"; Llama mean-pooled -1.92 and -0.96 vs last-token -10.12 and -12.62, verdict "readout". Duplicate shards cost about \$31.18.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_q2_pinned_layers](../2026-10-12_q2_pinned_layers/); [2026-10-12_q2_olmo_reader](../2026-10-12_q2_olmo_reader/); [2026-10-12_q2_llama_reader](../2026-10-12_q2_llama_reader/)

## Main scripts

- [`run_score_fix_mpolmo.py`](run_score_fix_mpolmo.py)
- [`run_score_fix_mpllama.py`](run_score_fix_mpllama.py)
- [`analyze_meanpool_reader.py`](analyze_meanpool_reader.py)
- [`analyze_shuffle_null_POSTHOC.py`](analyze_shuffle_null_POSTHOC.py)
- [`resume_llama_meanpool.py`](resume_llama_meanpool.py)

Result files the index names:

- `meanpool_reader_olmo_2026-10-12_212839.json` (not in this folder at the source commit)
- `meanpool_reader_llama_2026-10-13_012422.json` (not in this folder at the source commit)

## Files here

12 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q2_meanpool_readers/` at commit `2c551231d16dad4fa987f203d6a8172fc79dd880` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
