# 2026-10-14_q3_onpol_leakfix

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does a task leak across models inflate the results of the earlier phase's headline probes on Qwen3.5-27B's own runs? In the leak, frozen models trained on GPT-5.6-Sol's runs of a main task also score Qwen3.5-27B's runs of that same main task.

## What we did

Using stored scores only, kept for each Qwen3.5-27B run only the frozen cross-validated models whose training folds exclude its main task (8 of 20, plus a variant based on Sol's folds), then recomputed whether simulated catch matches real catch and the drop from on-policy to off-policy training.

## What we found

No conclusion changed: leak-free off-to-on drop for Qwen3-32B layer 48 at 5% −47.0 (95% −55.3 to −36.3) vs −45.4 published; no primary simulated-versus-real agreement verdict changed (readings_r2_2026-10-14.md; Amendment 5 to the fix-by-fix experiment's written rules).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_q3_onpolicy_titlecells](../2026-10-14_q3_onpolicy_titlecells/)

## Main scripts

- [`run_leakfix.py`](run_leakfix.py)
- [`run_leakfix_v2.py`](run_leakfix_v2.py)
- [`run_r2_leakfix.py`](run_r2_leakfix.py)

Result files the index names:

- `leakfix_2026-10-14_114543.json` (not copied: result left out to keep the release under 120 MB)
- `leakfix_v2_2026-10-14_115629.json` (not copied: result left out to keep the release under 120 MB)
- `r2_leakfix_2026-10-14_122641.json` (not copied: result left out to keep the release under 120 MB)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-14_q3_onpol_leakfix/` at commit `9e710f03864b257f10a653bf35a66cf995f872fe` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
