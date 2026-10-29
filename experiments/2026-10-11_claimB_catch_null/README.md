# 2026-10-11_claimB_catch_null

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Do the fixed-simulator test's catch rates at 1% and 5% false alarms (the share of simulated runs flagged) beat a label-shuffle null that shuffles within main tasks, rather than only a fixed 0.5?

## What we did

One Modal CPU container read stored probe-grid cells holding 100 task-grouped label shuffles each (the every-token readout grid, the grid on Qwen3.5-27B's own runs, the grid with development and test environments swapped, Qwen3.5-27B reading GPT-5.6-Sol's formatting-fixed runs, and Llama-3.2-1B) and recomputed catch at 1% and 5% for every shuffle, redoing the layer and regularization selection each time.

## What we found

PLAN.md result: 62 entries, column 0 reproduced all 50 published catches; the pre-registered end-of-output row of the experiment comparing last-token, mean-pooled and max-pooled output readouts (stand-in for the main Qwen3-32B end-of-output realism probe) caught 2.7% vs a null 95th percentile of 2.5%, p = 0.059, so not above its null at 1%; post hoc readouts, the Qwen3.5 reader (5.9%) and Llama-3.2-1B (7.8%) sit above their nulls; billed \$0.045.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-10_claimB_qwen35_reader](../2026-10-10_claimB_qwen35_reader/); [2026-10-10_claimB_onpolicy](../2026-10-10_claimB_onpolicy/); [2026-10-10_claimB_small_lm](../2026-10-10_claimB_small_lm/)

## Main scripts

- [`run_catch_null_modal.py`](run_catch_null_modal.py)

Result files the index names:

- [`catch_null_summary_2026-10-11_014446.json`](results/catch_null_summary_2026-10-11_014446.json)
- `catch_null_summary_v2_2026-10-11_014446.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_claimB_catch_null/` at commit `e6df21c7acd0d104c83e1482292fbcdf44aa610a` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
