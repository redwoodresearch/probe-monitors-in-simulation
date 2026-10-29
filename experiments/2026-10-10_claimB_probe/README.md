# 2026-10-10_claimB_probe

2026-10-10 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

For the fixed-simulator test (can probes tell simulated runs from real ones after the simulator's formatting mistakes are fixed?), does the probe-fitting code reproduce earlier results, and does the main probe's result hold when only converged fits are kept?

## What we did

Probe-fitting code for that test (building activation packs and grid-searching layer and regularization for the main end-of-output probe and for the every-token readout variant), a dry test on the earlier real-versus-simulated data, and a re-selection of the main probe from converged fits only, requested as an independent recheck.

## What we found

The dry run reproduces the earlier probe (0.917 with no text cleanup) with real run 2 vs run 1 at chance (0.516). With converged fits only, fixed-simulator vs real still scores 0.840 with no text cleanup and 0.822 with times and dates masked, and the first planted check (0.853) misses its 0.95 bar.

## In the post?

Not in the post.

## Status

Current (code; the result of the main Qwen3-32B end-of-output realism probe itself is reported in [2026-10-09_fixed_sim](../2026-10-09_fixed_sim/))

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/)

## Main scripts

- [`run_claimB_pack.py`](run_claimB_pack.py)
- [`run_claimB_grid.py`](run_claimB_grid.py)
- [`run_claimB_b_pack.py`](run_claimB_b_pack.py)
- [`run_claimB_b_grid.py`](run_claimB_b_grid.py)
- [`summarize_A_converged.py`](summarize_A_converged.py)

Result files the index names:

- `claimB_A_converged_2026-10-10_023335.json` (not in this folder at the source commit)
- `results_manifest_v2.json` (not in this folder at the source commit)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_claimB_probe/` at commit `7d44278055219878867a93d22795ea166cd97571` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
