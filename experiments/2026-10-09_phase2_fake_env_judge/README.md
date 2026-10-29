# 2026-10-09_phase2_fake_env_judge

2026-10-09 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

Does Qwen3.5-27B say in its own text that its environment looks fake or like a test, and does that differ between real and simulated runs?

## What we did

A cheap GPT-5-mini judge plus keyword regexes over Qwen3.5-27B's own messages and reasoning in the 140-run strict rerun, with planted turns as a positive control; later extended to all five 196-task launches with clustered intervals.

## What we found

The judge flagged 'looks fake' in 0 of 40 real runs, 2 of 39 September-simulated and 2 of 39 today's-simulated runs, and got 10 of 10 planted turns right. At 196 tasks judge-flagged rates were real 8.2%, September simulated 12.4%, second real attack 7.8% (not verified, groups not like for like).

## In the post?

Not in the post (related to the post's footnote on verbalized awareness, whose numbers come from the later [2026-10-14_q2_evalaware_step1](../2026-10-14_q2_evalaware_step1/)).

## Status

Current (preliminary)

## Builds on

[2026-10-07_phase2_attack_check](../2026-10-07_phase2_attack_check/) (strict rerun); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (196-task launches)

## Main scripts

- [`judge_fake_env.py`](judge_fake_env.py)
- [`rates_with_intervals.py`](rates_with_intervals.py)
- [`analyze_fake_env_clustered.py`](analyze_fake_env_clustered.py)

Result files the index names:

- [`fake_env_2026-10-10_002037.json`](results/fake_env_2026-10-10_002037.json)
- [`fake_env_2026-10-10_222140.json`](results/fake_env_2026-10-10_222140.json)
- [`clustered_rates_2026-10-10_222140.json`](results/clustered_rates_2026-10-10_222140.json)

## Files here

3 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-09_phase2_fake_env_judge/` at commit `2dc05afa611e03a9cd6110f93b971a336ec17421` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
