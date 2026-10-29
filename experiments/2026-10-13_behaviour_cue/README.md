# 2026-10-13_behaviour_cue

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

After every formatting fix, a probe still tells simulated runs from real ones; is the agent's own behaviour (what it did, not what it read) the rest of that giveaway?

## What we did

On honest runs only (GPT-5.6-Sol: 437 simulated honest runs against real honest runs on the same main tasks), computed per-run behaviour features by script (call counts, tool shares, tool bigrams, command lengths, repeat rate, first-edit position) plus reply-derived error flags, and fit a standardized logistic regression with 5-fold cross-validation grouped by main task, a 200-draw within-task label-shuffle null, and a real-run-1 vs real-run-2 control. \$0, local.

## What we found

behaviour_cue_2026-10-13_170635.json: behaviour features alone AUROC 0.635 (null 95th percentile 0.567); behaviour plus reply-derived error flags 0.859 (null 0.573); real-run-1 vs real-run-2 control 0.538 (null 0.537); ablation_POSTHOC: dropping the error-field feature leaves 0.631.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

None stated (cohorts la_public10_sim_astra_honest_n437 and la_public10_real_honest_n1000)

## Main scripts

- [`analyze_behaviour_cue.py`](analyze_behaviour_cue.py)
- [`check_POSTHOC_ablation.py`](check_POSTHOC_ablation.py)

Result files the index names:

- `behaviour_cue_2026-10-13_170635.json` (not copied: result left out to keep the release under 120 MB)
- [`ablation_POSTHOC_2026-10-13_171113.json`](results/ablation_POSTHOC_2026-10-13_171113.json)
- `s3_manifest_v1.json` (not copied: strings over 300 characters)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_behaviour_cue/` at commit `9f65915ef0236a37ed43a95911dffa68b1267bd1` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
