# 2026-10-13_claimB_author_split

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Which author carries the rest of the simulated-vs-real giveaway in honest runs after all formatting fixes: the simulator's replies (content) or GPT-5.6-Sol's commands (behaviour)?

## What we did

A text classifier (TF-IDF features, logistic regression, grouped label-shuffle null, real-vs-real control and positive control) run twice on the fully fixed simulated honest runs (437) vs real honest runs on the same tasks: once on simulator replies only, once on Sol's commands only; a rerun with a broader time mask. Modal CPU, about \$1.2 plus the rerun.

## What we found

Result JSONs: replies AUROC 0.741 (null 95th percentile 0.530; control 0.485); broader time mask 0.736; commands 0.539 (null 0.524; control 0.513). So the remaining giveaway sits mostly in the simulator's reply content, with only a faint behavioural trace.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/) (the fully fixed simulated honest arm sim-honest-fx-full)

## Main scripts

- [`run_author_split_modal.py`](run_author_split_modal.py)
- [`run_replies_broad_modal.py`](run_replies_broad_modal.py)
- [`analyze_POSTHOC_author_split.py`](analyze_POSTHOC_author_split.py)

Result files the index names:

- [`author_split_replies_POSTHOC_2026-10-13_173448.json`](results/author_split_replies_POSTHOC_2026-10-13_173448.json)
- [`author_split_commands_POSTHOC_2026-10-13_173449.json`](results/author_split_commands_POSTHOC_2026-10-13_173449.json)
- [`author_split_replies_broad_POSTHOC_2026-10-13_175920.json`](results/author_split_replies_broad_POSTHOC_2026-10-13_175920.json)
- [`author_split_replies_broad_l2_POSTHOC_2026-10-13_175920.json`](results/author_split_replies_broad_l2_POSTHOC_2026-10-13_175920.json)

## Files here

3 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-13_claimB_author_split/` at commit `ea0e8f319010c58cd9bd57c1afbfcc178c0a4a80` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
