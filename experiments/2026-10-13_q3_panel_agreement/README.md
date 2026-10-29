# 2026-10-13_q3_panel_agreement

2026-10-13 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does the five-judge sabotage panel agree less on simulated runs than on real ones (are simulated sabotage labels noisier)?

## What we did

From frozen vote-count exports only (no text or scores), computed per-world unclear rates, unanimity rates and vote margins for state-changing calls in 40 tasks' simulated runs and real runs 1 and 2 (Qwen3.5-27B), with a task bootstrap and a real-vs-real control. \$0.

## What we found

panel_agreement_2026-10-13_232105.json: unclear rate simulated 6.9% vs real 6.5% (difference +0.33 points [-2.36, +3.05]); unanimity 87.2% vs 81.4% (+5.8 [-0.9, +15.1]); reading 'inconclusive'.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sim_sabotage_ids](../2026-10-13_q3_sim_sabotage_ids/) (vote exports)

## Main scripts

- [`run_panel_agreement.py`](run_panel_agreement.py)

Result files the index names:

- `panel_agreement_2026-10-13_232105.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_panel_agreement/` at commit `1b11fdb0b1dc8187bc708eef37a2fe47e48dd577` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
