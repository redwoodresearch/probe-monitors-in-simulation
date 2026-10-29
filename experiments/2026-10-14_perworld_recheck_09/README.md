# 2026-10-14_perworld_recheck_09

2026-10-14 · part of [Checks of the main comparison](../../docs/experiments.md#checks-of-the-main-comparison)

## Question

Does an earlier result reproduce with independent code: that thresholding simulated attacks on simulated honest runs, instead of on real honest runs, reverses the "trap" of the mixed probe (trained on simulated attacks against real honest runs), which otherwise greatly overstates simulated catch?

## What we did

An independent recheck recomputed, from the same stored scores but with its own threshold, run-keeping and catch code and its own bootstrap seed, the simulated-minus-real gaps under both threshold conventions (one shared real threshold, or each setting's own) for the mixed probe (Qwen3-32B, layers 48 and 63) and for the earlier phase's headline Qwen3-32B probe (trained on simulated attacks against simulated honest runs), all on GPT-5.6-Sol's transcripts.

## What we found

Reproduced: Qwen3-32B layer 48 at 5%, shared-real gap +57.6 points (95% +52.7 to +61.4) vs per-world −13.6 (95% −19.6 to −4.3); simulated-honest false alarms fall from 58.1% to 4.9% at the shared vs own cutoff (perworld_recheck_2026-10-14_061650.printout.txt).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-14_q3_conventions](../2026-10-14_q3_conventions/)

## Main scripts

- [`recheck_perworld.py`](recheck_perworld.py)
- [`run_recheck_modal.py`](run_recheck_modal.py)

Result files the index names:

- [`perworld_recheck_2026-10-14_061650.json`](results/perworld_recheck_2026-10-14_061650.json)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_perworld_recheck_09/` at commit `28b4b661423ad8cb90a4b97ab092980c01daa34b` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
