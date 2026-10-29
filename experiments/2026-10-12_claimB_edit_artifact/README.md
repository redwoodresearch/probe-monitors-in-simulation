# 2026-10-12_claimB_edit_artifact

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

The project manager's artifact hypothesis: does the realism probe flag a simulated run more often when the formatting fixes edited more of its tool outputs?

## What we did

Post hoc, \$0: counted per simulated run how many tool outputs the four fix rules changed (editor insert and str_replace formatting, package-manager progress lines, and the clock shift), then compared the flag rate of the Qwen3-32B halves detector (train on one half of the environments and test on the other, in both directions) at the frozen 1% alarm in high-edit vs low-edit runs, with a length-matched comparison and a length-residualised rank correlation (2,000 task resamples).

## What we found

Codex post-check: high-edit runs were flagged less, 14/427 (3.3%) vs 37/512 (7.2%); length-matched difference -5.8 points [-9.2, -2.2]; residualised Spearman -0.087 [-0.137, -0.038]; reading "does not support" the hypothesis.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/); [2026-10-09_fixed_sim](../2026-10-09_fixed_sim/)

## Main scripts

- [`analyze_POSTHOC_edit_artifact.py`](analyze_POSTHOC_edit_artifact.py)

Result files the index names:

- `edit_artifact_POSTHOC_2026-10-12_114448.json` (not in this folder at the source commit)
- `results_manifest_v1.json` (not copied: JSON outside results/ and config/)
- `codex_post_2026-10-12.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_edit_artifact/` at commit `5fc3de2ef8376c3bd7751100a8026e61873d39c6` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
