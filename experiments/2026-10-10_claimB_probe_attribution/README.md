# 2026-10-10_claimB_probe_attribution

2026-10-10 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Which tool outputs and tokens carry the fixed-simulator probe's simulated-minus-real gap, and how much do the four leftover formatting slips explain?

## What we did

Free attribution of the frozen main Qwen3-32B end-of-output realism probe: per-output-type shares of the gap, overlap with leftover slips, rescoring with slip outputs dropped, and per-token projections, each gated by a planted check.

## What we found

Shell outputs carry about half the gap and the leftover slips 17%, but the rescore is not read because its planted check failed. The planted check failed (22% vs an 80% bar), so the breakdown (53% in shell outputs) is unvalidated.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-10_claimB_probe](../2026-10-10_claimB_probe/)

## Main scripts

- [`analyze_attribution_abc.py`](analyze_attribution_abc.py)
- [`run_attribution_tokens_modal.py`](run_attribution_tokens_modal.py)
- [`recheck_d_tail_modal.py`](recheck_d_tail_modal.py)
- [`analyze_plan_numbers.py`](analyze_plan_numbers.py)

Result files the index names:

- `results_manifest_v2.json` (not copied: JSON outside results/ and config/)
- `results_manifest_v6.json` (not copied: JSON outside results/ and config/)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_claimB_probe_attribution/` at commit `f3b4a08affce9c1f04cb8984fb122ebc1bcb4309` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
