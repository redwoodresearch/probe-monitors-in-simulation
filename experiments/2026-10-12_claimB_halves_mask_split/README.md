# 2026-10-12_claimB_halves_mask_split

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Which of the two masking steps, numbers and IDs or whitespace, removes the halves gap (train on one half of the environments and test on the other, in both directions; simulated versus second-real flag rates at the 1% threshold)?

## What we did

Re-read and refit the Qwen3-32B halves test with only numbers and IDs masked or only whitespace normalised in both worlds; scored simulated and second-real runs at the frozen 1% alarm with converged-only selection, refit label-shuffle nulls and planted checks.

## What we found

RESULT.md: numbers and IDs masked +4.90 points [+2.19, +7.29] ("survives"); whitespace normalised +0.75 [-0.53, +2.17] ("no excess shown"), change vs unmasked halves -3.09 [-6.56, -0.87]; all planted checks pass.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/); [2026-10-12_claimB_halves_mask23](../2026-10-12_claimB_halves_mask23/)

## Main scripts

- [`make_result_md.py`](make_result_md.py) (runners as run, on a side branch)

Result files the index names:

- `halves_splitN_2026-10-12_143658.json` (not in this folder at the source commit)
- `halves_splitW_2026-10-12_143656.json` (not in this folder at the source commit)
- `RESULT.md` (not copied: notes or README (internal working labels))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_halves_mask_split/` at commit `e01254926c328276f77a8f68bc88e1fab4e5deb2` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
