# 2026-10-11_claimB_maxwin_rescore

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does re-scoring the fixed-simulator test's realism probe with the maximum of rolling means (about 10-token windows) instead of each output's last token let it catch the planted "this environment is simulated" line it missed, and raise catch at a strict alarm?

## What we did

Re-scored the frozen probe on Modal using stored per-token Qwen3-32B readings that exist only for a subset of runs (the cheap exploratory option; a full GPU re-read was to follow only if the planted line was caught): output-end mean versus 10-token rolling-mean maximum, on the planted-line pairs and the test runs, plus a curve of the score by token offset from the output's end.

## What we found

Commit aa3bb9c2a: the frozen probe reads the planted line at chance (output-end 0.495, window 10 0.472), so option (b) was not triggered; on 100 simulated-vs-real test pairs output-end 0.835 vs window-10 maximum 0.483 (PLAN.md result, option (a)); the score peaks at the last content token (0.814) and decays to 0.61 one token earlier; catch values are in-sample, descriptive only; the first launch crash-looped.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (its first probe step); [2026-10-10_claimB_probe](../2026-10-10_claimB_probe/)

## Main scripts

- [`run_a_subset_modal.py`](run_a_subset_modal.py)
- [`analyze_power_maxwin.py`](analyze_power_maxwin.py)
- [`analyze_outputs_per_run.py`](analyze_outputs_per_run.py)

Result files the index names:

- `maxwin_a_2026-10-11_211939.json` (not in this folder at the source commit)
- `power_maxwin_2026-10-11_194043.json` (not in this folder at the source commit)
- `results_manifest_v1.json` (not copied: JSON outside results/ and config/)
- `codex_factcheck_maxwin_a_2026-10-11.txt` (not copied: notes or logs outside results/)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_maxwin_rescore/` at commit `2ed8fa88cf0fac86b0da8431a33c344968bd5d02` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
