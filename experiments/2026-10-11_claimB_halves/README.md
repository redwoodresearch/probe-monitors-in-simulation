# 2026-10-11_claimB_halves

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

After all four formatting fixes, does a probe trained on half the environments still tell fixed-simulator runs from real ones on the other half, beyond the gap between two real runs?

## What we did

Qwen3-32B activations with all four fixes; the main Qwen3-32B end-of-output realism probe (linear) trained on the 5 dev environments and scored on the 5 test environments and the reverse; catch at a 1-in-100 alarm set on held-out real run 1, with a planted check and real run 2 as the yardstick; post hoc direct comparison scoring real run 2 with the same detector.

## What we found

Yes in both directions; pooled 5.4% of simulated runs vs 0.5% of real run 2 flagged (gap +4.9 points), AUROC 0.743 and 0.729 vs 0.490 and 0.476 (commit 7fed05c94). Post hoc: same detector flags simulated 5.4% vs real run 2 1.6% (+3.8 [+1.9, +7.0]).

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_claimB_h3_fix_to_plateau](../2026-10-11_claimB_h3_fix_to_plateau/) (test-environment reads); [2026-10-11_claimB_slips_evened_probe](../2026-10-11_claimB_slips_evened_probe/) (dev-environment reads)

## Main scripts

- [`run_halves_modal.py`](run_halves_modal.py)
- [`analyze_halves.py`](analyze_halves.py)
- [`analyze_power_halves.py`](analyze_power_halves.py)
- [`run_POSTHOC_halves_e2_modal.py`](run_POSTHOC_halves_e2_modal.py)
- [`analyze_POSTHOC_halves_e2.py`](analyze_POSTHOC_halves_e2.py)

Result files the index names:

- `halves_2026-10-11_164208.json` (not copied: result left out to keep the release under 120 MB)
- `halves_2026-10-11_164208.png` (not copied: result file type not copied (.png))
- `halves_e2_direct_POSTHOC_2026-10-11_172057.json` (not copied: result left out to keep the release under 120 MB)
- `halves_e2_direct_POSTHOC_2026-10-11_172057.png` (not copied: result file type not copied (.png))
- `power_halves_2026-10-11_153819.json` (not copied: result left out to keep the release under 120 MB)

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_halves/` at commit `53da7f48ee6919e31d8d1049b202c1d0454a3cf0` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
