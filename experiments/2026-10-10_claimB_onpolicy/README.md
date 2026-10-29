# 2026-10-10_claimB_onpolicy

2026-10-10 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

The fixed-simulator test on policy: can a text classifier and a probe tell Qwen3.5-27B's simulated September attack runs (with the frozen formatting fixes) from its real ones, beyond two real runs of the same task?

## What we did

Applied the frozen fix rules once to 190 complete task triples from the 196-task scale-up (real attack run 1, run 2, simulated), then ran Qwen3.5-27B probes and TF-IDF on five held-out test environments with real-versus-real controls; amendment 1 swapped development and test environments; post hoc catch analyses.

## What we found

Weak difference by the pre-set rule, 0.561 [0.506, 0.615] vs 0.465 for real versus real, but both planted checks fail. Separation transfers across the environment swap (0.623, 0.660) but catch at 1% only 2.5% and 5.7%. Thread closed.

## In the post?

Not in the post.

## Status

Closed; amendment 1 plan row superseded by its run

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (fix rules); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (196-task runs and extractions)

## Main scripts

- [`build_reader_inputs.py`](build_reader_inputs.py)
- [`stage_probe_inputs.py`](stage_probe_inputs.py)
- [`run_onpolicy_pack.py`](run_onpolicy_pack.py)
- [`run_onpolicy_scores_modal.py`](run_onpolicy_scores_modal.py)
- [`run_tfidf_onpolicy_modal.py`](run_tfidf_onpolicy_modal.py)
- [`analyze_onpolicy_prefix.py`](analyze_onpolicy_prefix.py)
- [`analyze_POSTHOC_output_catch.py`](analyze_POSTHOC_output_catch.py)

Result files the index names:

- `results_manifest_v1.json` (not copied: JSON outside results/ and config/)
- `results_manifest_v2.json` (not copied: JSON outside results/ and config/)
- `build_counts_2026-10-10_153909.json` (not in this folder at the source commit)
- `claimB_b_grid_2026-10-10_213914.json` (not in this folder at the source commit)

## Files here

11 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_claimB_onpolicy/` at commit `416c544d2850e7086c4489b7195a385ac3d16895` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
