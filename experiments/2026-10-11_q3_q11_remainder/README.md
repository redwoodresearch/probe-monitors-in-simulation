# 2026-10-11_q3_q11_remainder

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

After the four slips are evened, where does the frozen probe's remaining fixed-simulator-versus-real gap sit, and do simple structural features of the shell replies explain it?

## What we did

Post hoc per-output rescore of the frozen layer-63 probe on stored Qwen3-32B activations (Modal), splitting the gap by output type; then 19 structural features of bash replies (length, lines, blanks, error markers, trailing newline and so on) chosen on one task half and tested on the other.

## What we found

README: 77% [71, 84] of the remaining gap (0.244) sits in shell replies; no simple structural feature explains it (best three features 1.1% [-0.3, 2.7], against a pre-registered 50% bar). Our experiment log cites the 2.8% figure for the 19 layout features.

## In the post?

Not in the post.

## Status

Exploratory (post hoc)

## Builds on

[2026-10-11_claimB_slips_evened_probe](../2026-10-11_claimB_slips_evened_probe/)

## Main scripts

- [`run_q11_rescore_modal.py`](run_q11_rescore_modal.py)
- [`analyze_q11_remainder.py`](analyze_q11_remainder.py)
- [`plot_q11_remainder.py`](plot_q11_remainder.py)
- [`run_q11_features_modal.py`](run_q11_features_modal.py)
- [`analyze_q11_features.py`](analyze_q11_features.py)
- [`plot_q11_features.py`](plot_q11_features.py)

Result files the index names:

- `q11_remainder_POSTHOC_2026-10-11_023527.json` (not copied: result left out to keep the release under 120 MB)
- [`q11_rescore_gate_POSTHOC_2026-10-11_023512.json`](results/q11_rescore_gate_POSTHOC_2026-10-11_023512.json)
- `q11_features_POSTHOC_2026-10-11_100359.json` (not copied: result left out to keep the release under 120 MB)
- [`q11_features_run_POSTHOC_2026-10-11_100245.json`](results/q11_features_run_POSTHOC_2026-10-11_100245.json)
- `q11_remainder_POSTHOC_2026-10-11_023527.png` (not copied: figure (not copied))
- `q11_features_POSTHOC_2026-10-11_100359.png` (not copied: figure (not copied))

## Files here

6 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-11_q3_q11_remainder/` at commit `f7ea699bc6df02cefad8f03391f6faeae3314733` of our internal repository. 8 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
