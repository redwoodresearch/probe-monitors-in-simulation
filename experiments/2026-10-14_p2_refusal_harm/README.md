# 2026-10-14_p2_refusal_harm

2026-10-14 · part of [Training and scoring the probes](../../docs/experiments.md#training-and-scoring-the-probes)

## Question

Fit the published harmfulness and refusal directions (difference of mean activations on harmful vs harmless prompts) on Qwen3.5-27B, check them, and score Qwen3.5's runs.

## What we did

Read andyrdt/refusal_direction's public prompt splits on an H100, chose the layer per direction on validation data (harmfulness at the request's last token, refusal at the last prompt token), passed a positive-control gate, scored the 196-task corrected-render runs, then refit on H200 and compared.

## What we found

Harmfulness layer 34 and refusal layer 47, validation and test AUROC 1.000, gate passed (read_fit printout); H200 refit cosine 0.99999 to the first fit, catches differing by up to 1.1 points (h200_compare printout).

## In the post?

In the post: Table 1, Table 2 and Appendix A (harmfulness and refusal directions)

## Status

Current

## Builds on

Nothing earlier in this index.

## Main scripts

- [`run_read_fit.py`](run_read_fit.py)
- [`run_score_linear.py`](run_score_linear.py)
- [`run_h200_check.py`](run_h200_check.py)

Result files the index names:

- `read_fit_2026-10-14_125514.json` (not in this folder at the source commit)
- [`h200_check_2026-10-14_145255.json`](results/h200_check_2026-10-14_145255.json)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 1 result file the index names, they match in 3 script-and-commit pairs.

## Files here

3 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-14_p2_refusal_harm/` at commit `d2d1e0e76810656b3a5367825e3ec8a3b9eede43` of our internal repository. 15 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
