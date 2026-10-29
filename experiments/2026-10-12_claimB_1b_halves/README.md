# 2026-10-12_claimB_1b_halves

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does a small language model (Llama-3.2-1B) pick up the same simulated-versus-real difference as the Qwen3-32B realism probe in the environment-halves test, i.e. could generic text statistics explain the probe flagging 5.4% of simulated runs vs 1.6% of second real runs?

## What we did

Read the 1,783 evened (formatting-fixed) runs with Llama-3.2-1B-Instruct on Modal, merged them with stored Llama rows, and refit the halves recipe (train on half of the environments, test on the other half, both directions; Llama layers 14 and 15; converged fits only), scoring simulated runs and a second real run at the frozen 1-in-100 alarm, with the planted-slip gate.

## What we found

Codex fact-check file: Llama flags 36/939 simulated vs 10/939 second-real runs (+2.77 points [+0.43, +4.23]); the probe flags 51/939 vs 15/939 (+3.83 [+1.92, +7.05]); paired Llama minus probe -1.06 [-4.98, +1.05]; both Llama planted gates pass. Descriptive, no verdict.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/); [2026-10-10_claimB_small_lm](../2026-10-10_claimB_small_lm/); [2026-10-11_claimB_halves_qwen35](../2026-10-11_claimB_halves_qwen35/); [2026-10-12_claimB_tfidf_halves](../2026-10-12_claimB_tfidf_halves/)

## Main scripts

- [`run_1b_read_modal.py`](run_1b_read_modal.py)
- [`run_1b_refit_modal.py`](run_1b_refit_modal.py)
- [`analyze_1b_halves.py`](analyze_1b_halves.py)
- [`price_1b_halves.py`](price_1b_halves.py)

Result files the index names:

- `1b_halves_2026-10-12_130117.json` (not in this folder at the source commit)
- `results_manifest_v1.json` (not copied: JSON outside results/ and config/)
- `codex_factcheck_1b_halves_2026-10-12_130117.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `price_2026-10-12_115323.json` (not copied: result left out to keep the release under 120 MB)

## Files here

6 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_1b_halves/` at commit `dbb0486f41e2ff28bcf13a65feb6e604265e5026` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
