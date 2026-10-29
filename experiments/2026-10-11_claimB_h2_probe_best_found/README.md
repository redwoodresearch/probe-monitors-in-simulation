# 2026-10-11_claimB_h2_probe_best_found

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

How much better than the main Qwen3-32B end-of-output realism probe (linear) can a richer probe separate fixed-simulator runs from real ones on the stored activations?

## What we did

Searched 12 probe configurations (linear and nonlinear; end-of-output, mean, moving-average, max and call-end readouts across Qwen3-32B layers) on stored activations of 939 task cells (text with times and dates masked), with a shuffle null that repeats the whole search, plus a dev-environment check; Modal.

## What we found

Best-found probe 0.841 vs 0.822, inconclusive against the 0.03 bar, and its 1% catch lead (19.7% vs 2.7%) did not replicate. Catch at a 1% target is 2.7% on test environments vs about 23% on practice ones, an environment difference.

## In the post?

Not in the post.

## Status

Exploratory (closed as too few)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-10_claimB_probe](../2026-10-10_claimB_probe/) (activation packs)

## Main scripts

- [`run_h2_search_modal.py`](run_h2_search_modal.py)
- [`analyze_h2_search.py`](analyze_h2_search.py)
- [`plot_h2_search.py`](plot_h2_search.py)
- [`run_h2_devcheck_modal.py`](run_h2_devcheck_modal.py)
- [`analyze_h2_devcheck.py`](analyze_h2_devcheck.py)
- [`analyze_POSTHOC_h2_catch_by_env.py`](analyze_POSTHOC_h2_catch_by_env.py)
- [`analyze_POSTHOC_h2_catch_null.py`](analyze_POSTHOC_h2_catch_null.py)

Result files the index names:

- `h2_summary_2026-10-11_132926.png` (not copied: result file type not copied (.png))
- `h2_devcheck_2026-10-11_144025.png` (not copied: result file type not copied (.png))
- `plant1_layers_POSTHOC_2026-10-11_160035.json` (not copied: result left out to keep the release under 120 MB)
- [`summary_keys_POSTHOC_2026-10-12_102853.json`](results/summary_keys_POSTHOC_2026-10-12_102853.json)

## Files here

11 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-11_claimB_h2_probe_best_found/` at commit `c1a30e7e94a94d061e6e24d91f17f671a3dda661` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
