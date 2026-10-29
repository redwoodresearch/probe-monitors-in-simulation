# 2026-10-12_claimB_halves_errors

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does the halves gap (train on one half of the environments and test on the other, in both directions; 5.4% of simulated vs 1.6% of second real runs flagged at the 1% threshold) come from tasks where the real run hit a tool error that the simulator hid?

## What we did

Post hoc, \$0: split the 939 halves cells by whether the first real run had a recorded structured tool error and recomputed flag rates (simulated vs second real run) in each group from stored Qwen3-32B halves scores, with the planted check and two-real-runs yardstick per group.

## What we found

RESULT.md: "persists without errors": cells without errors +2.86 points [+0.59, +6.12] (20/489 vs 6/489); cells with errors +4.89 [+2.10, +9.07]; difference +2.03 [-1.53, +5.72], so the gap is not shown to concentrate in error cells.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/); [2026-10-07_sim_failure_hiding](../2026-10-07_sim_failure_hiding/)

## Main scripts

- [`analyze_POSTHOC_hidden_errors.py`](analyze_POSTHOC_hidden_errors.py)
- [`make_result_md.py`](make_result_md.py)

Result files the index names:

- `hidden_errors_POSTHOC_2026-10-12_102409.json` (not in this folder at the source commit)
- `RESULT.md` (not copied: notes or README (internal working labels))
- `codex_factcheck_hidden_errors_POSTHOC_2026-10-12_102409.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_halves_errors/` at commit `54beafa34300b9cad2faa3685ffef70e0fe9ff09` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
