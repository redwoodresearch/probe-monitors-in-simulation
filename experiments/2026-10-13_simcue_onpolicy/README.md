# 2026-10-13_simcue_onpolicy

2026-10-13 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Does a frozen simulation-cue probe (trained earlier to detect the model registering that it is in a simulation) fire more on Qwen3.5-27B's own simulated runs than on its real runs?

## What we did

Applied the frozen layer-16 token-max simulation-cue probe, trained on matched contrasts inside real honest transcripts, to Qwen3.5-27B's staged on-policy replays (first and second real attack, September simulated attack, two honest sets; about 196 tasks), with a threshold flagging 1% of first-real runs, a shuffled-label null and post hoc argmax-span checks. Modal, under \$10.

## What we found

analysis_2026-10-13_124152.json: at the 1% threshold, 2 of 191 simulated runs (1.0%) are above it against 1 of 194 for both real runs; at the frozen threshold of the simulation-cue probe trained on matched contrasts inside real honest transcripts, no run of any world is flagged. The cue probe does not pick out simulated runs.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_matched_training_d](../2026-10-12_matched_training_d/) (matched-training probe)

## Main scripts

- [`run_simcue_onpolicy.py`](run_simcue_onpolicy.py)
- [`analyze_simcue_onpolicy.py`](analyze_simcue_onpolicy.py)
- [`check_POSTHOC_argmax_markup.py`](check_POSTHOC_argmax_markup.py)
- [`check_POSTHOC_argmax_spans.py`](check_POSTHOC_argmax_spans.py)

Result files the index names:

- [`analysis_2026-10-13_124152.json`](results/analysis_2026-10-13_124152.json)
- [`score_2026-10-13_121721_score.json`](results/score_2026-10-13_121721_score.json)
- [`check_2026-10-13_121629_check.json`](results/check_2026-10-13_121629_check.json)
- `argmax_spans_POSTHOC_2026-10-13_144608.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- [`argmax_markup_POSTHOC_2026-10-13_142742.json`](results/argmax_markup_POSTHOC_2026-10-13_142742.json)

## Files here

4 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-13_simcue_onpolicy/` at commit `f1c6b658a591a625daf2a943bfb09063f9fc41e6` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
