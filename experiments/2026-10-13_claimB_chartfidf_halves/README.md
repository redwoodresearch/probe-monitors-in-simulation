# 2026-10-13_claimB_chartfidf_halves

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

In the environment-halves realism test (training on one half of the environments and testing on the other, in both directions: does a simulation detector flag fixed simulated runs more than a second real run at the frozen 1% threshold?), is the gap between the Qwen3-32B probe and a word TF-IDF text baseline due to whitespace that a word tokenizer drops?

## What we did

Reran the environment-halves TF-IDF baseline (training on one half of the environments and testing on the other, in both directions) with character-only n-gram vectorizers that keep whitespace (n = 1 to 5 and 2 to 5), on the same halves, folds and frozen 1% alarm, plus the whitespace-masked arm, with a paired-draw interval against the word-plus-character TF-IDF. Modal CPU, \$0.34.

## What we found

chartfidf_halves_POSTHOC_2026-10-13_004947.json, pooled simulated-minus-second-real gap at 1%: word plus character +1.81 points [+0.75, +4.22]; character 1-5 +2.66 [+1.29, +5.08]; character 2-5 +2.56 [+1.15, +4.86]; character 1-5 on whitespace-masked text +1.28 [+0.21, +3.33]; character 2-5 minus word plus character +0.75 [-0.73, +1.52]. Character models flag simulated runs slightly more, but the design cannot isolate dilution.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

2026-10-12 halves design (tfidf_halves_POSTHOC @ faafd24c; experiments/[2026-10-06_q1_text_baselines](../2026-10-06_q1_text_baselines/)/analyze_tfidf.py)

## Main scripts

- [`run_chartfidf_halves_modal.py`](run_chartfidf_halves_modal.py)
- [`analyze_chartfidf_halves.py`](analyze_chartfidf_halves.py)

Result files the index names:

- `chartfidf_halves_POSTHOC_2026-10-13_004947.json` (not copied: result left out to keep the release under 120 MB)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_claimB_chartfidf_halves/` at commit `f0b39ff266cd2772278acbc99685af92c3aec4ca` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
