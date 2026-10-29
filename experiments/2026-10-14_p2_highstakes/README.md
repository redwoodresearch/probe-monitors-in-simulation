# 2026-10-14_p2_highstakes

2026-10-14 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Fit the high-stakes probe of McKenzie et al. (arXiv 2506.10805) on Qwen3.5-27B from their public data, check it, and score Qwen3.5's runs.

## What we did

Read the public "models under pressure" data on two H200 shards, fit logistic regressions per layer and chose layer 43 on held-out data, certified convergence, then scored the 196-task corrected-render runs.

## What we found

Layer 43, held-out AUROC 0.999, test 0.999, the separate balanced Anthropic helpful/harmless test set 0.886, gate passed (read_fit printout); tight refit certified and picks the same layer (convergence printout); pilot reply-readout attack-vs-honest AUROC 0.582 (scores printout).

## In the post?

In the post: Table 1, Table 2 and Appendix A (high-stakes probe)

## Status

Current

## Builds on

Nothing earlier in this index.

## Main scripts

- [`run_read_fit.py`](run_read_fit.py)
- [`run_convergence.py`](run_convergence.py)

Result files the index names:

- `read_fit_2026-10-14_131233.json` (not in this folder at the source commit)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

2 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-14_p2_highstakes/` at commit `d2d1e0e76810656b3a5367825e3ec8a3b9eede43` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
