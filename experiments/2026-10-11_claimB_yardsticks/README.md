# 2026-10-11_claimB_yardsticks

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Fixed-simulator test yardsticks: does the probe beat a punctuation-and-whitespace-only text baseline, and are two simulated rerolls of the same task as alike as two real runs?

## What we did

Fitted a character-skeleton classifier on the text of the 946 test runs with times and dates masked, and simulated a second reroll of 97 test cells with the fixed simulator, read with Qwen3-32B on Modal, scoring both with the frozen probe; exploratory.

## What we found

README: probe minus skeleton +0.269 [+0.235, +0.304], but the skeleton fails both planted checks (0.50, 0.52); reroll yardstick: simulated-vs-simulated is as alike as two real runs at both layers (layer 63: D = -0.196, upper bound -0.124 below 0.076, where D = AUROC for distinguishing two simulated rerolls minus AUROC for distinguishing two real runs), but underpowered (about 27%) and the real-vs-real draw is unusual.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (its first probe step)

## Main scripts

- [`run_skeleton_modal.py`](run_skeleton_modal.py)
- [`analyze_skeleton.py`](analyze_skeleton.py)
- [`stage_reroll_upload.py`](stage_reroll_upload.py)
- [`run_reroll_modal.py`](run_reroll_modal.py)
- [`analyze_reroll.py`](analyze_reroll.py)
- [`analyze_POSTHOC_reroll_orientation.py`](analyze_POSTHOC_reroll_orientation.py)
- [`analyze_POSTHOC_reroll_subsets.py`](analyze_POSTHOC_reroll_subsets.py)

Result files the index names:

- `skeleton_vs_probe_2026-10-11_230815.json` (not copied: result left out to keep the release under 120 MB)
- `reroll_yardstick_2026-10-11_234022.json` (not in this folder at the source commit)
- `reroll_orientation_POSTHOC_2026-10-12_014022.json` (not copied: result left out to keep the release under 120 MB)
- `reroll_headers_2026-10-11_224554.json` (not in this folder at the source commit)
- `codex_factcheck_skeleton_2026-10-12.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `codex_factcheck_reroll_2026-10-12.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

10 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_claimB_yardsticks/` at commit `aa514776e1c788fc5c6d2843399d69d498c91e95` of our internal repository. 18 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
