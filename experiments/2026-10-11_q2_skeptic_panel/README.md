# 2026-10-11_q2_skeptic_panel

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Is each settled line on our internal status page supported by the numbers and checks the page itself gives? (a skeptic panel of three models)

## What we did

Three models from three labs (GPT-6-Astra, Gemini 3.1 Pro, Claude Opus 5.5) on OpenRouter, pinned to first-party providers, judged each status-page question's settled line, naming the likeliest overclaim and the cheapest check that could change it.

## What we found

README: 45 of 45 calls returned valid answers (\$0.59); verdicts were mostly 'yes' or 'partly' per question (for example on the detectability of formatting-fixed simulations when Qwen3.5-27B acts, and on whether the probe's real-versus-simulated gap is larger when Qwen acts than when it reads: 'partly' from all three); the question of why the probe beats the text classifier after simulator fixes was rerun alone on version 470.

## In the post?

Not in the post.

## Status

Current

## Builds on

Answer lines of our internal status page; [2026-10-11_claims_ledger](../2026-10-11_claims_ledger/)

## Main scripts

- [`build_panel_input.py`](build_panel_input.py)
- [`run_panel.py`](run_panel.py)

Result files the index names:

- `panel_summary_2026-10-11_032841.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `panel_summary_2026-10-11_034524.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `panel_summary_2026-10-11_041320.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `panel_input_v468.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_q2_skeptic_panel/` at commit `4f43df3971bda7d30caad9eb221f6e4063dc1db9` of our internal repository. 19 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
