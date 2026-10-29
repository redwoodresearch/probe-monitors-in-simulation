# 2026-10-11_q2_deception_blackbox

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does the frozen deception probe (Apollo's recipe retrained on Qwen3.5-27B) rank attack-instructed runs above honest runs better than a black-box judge reading the same commands?

## What we did

Built judge inputs from the exact commands the probe scored, ran an LLM judge (with the project lead's OK to read attack commands, \$90 stop) on 683 runs over 171 tasks, and compared within-task AUROC with the probe, with real run 2 as yardstick and a planted check.

## What we found

README: no; the judge does at least as well, probe 0.768 vs judge 0.943 (difference -0.174 [-0.239, -0.110]); run 2 gives -0.152; the replication rule failed on precision; plant detected.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-10_q2_deception_runlevel](../2026-10-10_q2_deception_runlevel/); [2026-10-09_q2_deception_apollo](../2026-10-09_q2_deception_apollo/)

## Main scripts

- [`build_judge_inputs.py`](build_judge_inputs.py)
- [`run_judges.py`](run_judges.py)
- [`analyze_blackbox.py`](analyze_blackbox.py)
- [`check_tokens.py`](check_tokens.py)
- [`count_attempt_labels.py`](count_attempt_labels.py)
- [`check_readme_claims.py`](check_readme_claims.py)

Result files the index names:

- `blackbox_2026-10-11_235040.json` (not copied: result left out to keep the release under 120 MB)
- `attempt_labels_2026-10-12_000325.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `judge_inputs_summary_2026-10-11_233130.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `readme_claims_2026-10-12_031639.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_q2_deception_blackbox/` at commit `67f4991dbab1f56077dd4950404968aa437332e2` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
