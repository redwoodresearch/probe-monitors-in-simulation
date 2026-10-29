# 2026-10-12_claimB_halves_number_shuffle

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does any single whitespace operation (line-final whitespace, runs of blank lines, trailing whitespace) remove the halves gap (train on one half of the environments and test on the other, in both directions; simulated versus second-real flag rates at the 1% threshold)? (Number-shuffle arms were planned and priced but never run.)

## What we did

Priced number-shuffle arms and counted staging; amendment 2 then ran only three whitespace arms, each operation alone in both worlds (removing line-final whitespace except the newline; replacing runs of three or more newlines with two; removing all whitespace at the text's end), through the mask-split design on Qwen3-32B: re-read, refit, frozen 1% alarm, planted checks.

## What we found

RESULT.md (at 7cd4f6ab): all three "survive": removing line-final whitespace except the newline +3.09 points [+1.58, +6.08], replacing runs of three or more newlines with two +2.56 [+0.98, +5.86], removing all whitespace at the text's end +2.45 [+0.32, +4.52]. All planted checks pass. Each single-operation point estimate leaves a larger gap than all three operations together, but the intervals for the second and the third operation minus all three together include zero. These comparisons are descriptive, exploratory and preliminary, with intervals unadjusted across arms.

## In the post?

Not in the post.

## Status

Exploratory (preliminary; intervals unadjusted across arms); number-shuffle arms never ran

## Builds on

[2026-10-12_claimB_halves_mask_split](../2026-10-12_claimB_halves_mask_split/)

## Main scripts

- `run_split_read_W1/W2/W3_modal.py` (not in this folder at the source commit)
- `run_split_refit_W1/W2/W3_modal.py` (not in this folder at the source commit)
- [`analyze_split.py`](analyze_split.py)
- [`count_staging_modal.py`](count_staging_modal.py) (on a side branch
- the latest ref holds only watchers)

Result files the index names:

- `halves_splitW1_2026-10-12_172121.json` (not in this folder at the source commit)
- `halves_splitW2_2026-10-12_172010.json` (not in this folder at the source commit)
- `halves_splitW3_2026-10-12_172329.json` (not in this folder at the source commit)
- `staging_counts_2026-10-12_152618.json` (not in this folder at the source commit)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_halves_number_shuffle/` at commit `23a4383b87be94f271082e3ff0451db880b0abc4` of our internal repository. Every file in that folder was copied. We also copied 2 main scripts that the index names but that commit lacks, each from its own latest commit on another branch; the manifest gives each one's source commit.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
