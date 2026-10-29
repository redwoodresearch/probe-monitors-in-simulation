# 2026-10-11_claimB_k_replies

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

What carries the unexplained half of the fixed-simulator probe's gap: one simulated reply itself, or simulated history building up? (k-simulated-replies test, k = 1 stage)

## What we did

Kept real run histories and had the fixed simulator write only the last reply at 400 selected sites (GPT-6-Astra), read them with Qwen3-32B on Modal, and measured how much of the full simulated-minus-real probe gap one reply gives, with a planted slip and a self-swap check; larger k (2, 4, 8) was only power-planned.

## What we found

One swapped reply gives about a quarter of the full gap (0.250 [0.141, 0.382], below the 0.5 bar), so not the reply itself; checks passed; 198 of 400 replies were identical after the fixes. Post hoc: where the reply differed (202 of 400) it carried 0.529 [0.322, 0.812].

## In the post?

Not in the post.

## Status

Exploratory (preliminary; follow-up [2026-10-11_claimB_swap_changed](../2026-10-11_claimB_swap_changed/) closed)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/); [2026-10-11_claimB_h1_text_vs_layout](../2026-10-11_claimB_h1_text_vs_layout/) (swap runner imported)

## Main scripts

- [`select_k1_sites.py`](select_k1_sites.py)
- [`run_k1_simulate.py`](run_k1_simulate.py)
- [`build_k1_records.py`](build_k1_records.py)
- [`run_k1_modal.py`](run_k1_modal.py)
- [`analyze_k1.py`](analyze_k1.py)
- [`analyze_POSTHOC_k1_identical.py`](analyze_POSTHOC_k1_identical.py)
- [`analyze_power_k_replies.py`](analyze_power_k_replies.py)
- [`analyze_POWER_k8.py`](analyze_POWER_k8.py)

Result files the index names:

- [`k1_summary_2026-10-11_174133.json`](results/k1_summary_2026-10-11_174133.json)
- `k1_summary_2026-10-11_174133.png` (not copied: result file type not copied (.png))
- `k1_identical_POSTHOC_2026-10-11_174846.json` (not copied: result left out to keep the release under 120 MB)
- `k1_identical_source_POSTHOC_2026-10-11_194340_summary.json` (not copied: result left out to keep the release under 120 MB)
- `k1_check_only_2026-10-11_165703.json` (not copied: result left out to keep the release under 120 MB)
- [`k1_sim_ledger_2026-10-11_170940.json`](results/k1_sim_ledger_2026-10-11_170940.json)
- `attachment_counts_2026-10-11_165355.json` (not copied: result left out to keep the release under 120 MB)

## Files here

13 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-11_claimB_k_replies/` at commit `1d8bd5a32b1bd16cc5a675ff77e7311b258435f3` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
