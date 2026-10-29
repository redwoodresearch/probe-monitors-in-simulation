# 2026-10-11_is_this_a_test

2026-10-11 · part of [Side question: does the model notice it is being simulated or evaluated?](../../docs/experiments.md#side-question-does-the-model-notice-it-is-being-simulated-or-evaluated)

## Question

"Is this a test?": asked directly whether the environment was real or simulated, does a reader model answer differently for simulated and real runs, and can a probe for the meaning of simulation notices be validated?

## What we did

Part 1: Qwen3-32B read each run of the fixed-simulator test (939 main-task/side-task cells) and was asked "real or simulated?", with the probability of "simulated" as the readout and planted notice lines as gates; part 2: probes for simulation notices checked against pre-set gates; a positive control on a known-weak simulator (from a separate 18-environment LinuxArena cohort); then a control step (step 1) asking Qwen3.5-27B the same question about its own attack runs from the September simulator.

## What we found

README: part 1 within the margins of two real runs (rank advantage +0.003 [-0.021, +0.027], R (the mean absolute simulated-to-real score distance, averaged over both real runs, divided by the mean absolute distance between the two real runs) 0.929 [0.883, 0.985]); part 2 does not validate (fails neutral and reading-position gates); the readout does see the weak simulator (+0.313 [+0.258, +0.369]), possibly via length; ask35_control_verdict: Qwen3.5 step 1 'fail', so step 2 did not launch.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (fixed-simulator test runs); [2026-10-11_claimB_weaksim_control](../2026-10-11_claimB_weaksim_control/) (the 18-environment set); [2026-10-11_q2_e1_sentences](../2026-10-11_q2_e1_sentences/) (design)

## Main scripts

- [`run_ask_qwen.py`](run_ask_qwen.py)
- [`analyze_part1.py`](analyze_part1.py)
- [`run_pairs.py`](run_pairs.py)
- [`run_pairs2.py`](run_pairs2.py)
- [`spawn_fit_itat.py`](spawn_fit_itat.py)
- [`analyze_part2_gates.py`](analyze_part2_gates.py)
- [`run_ask_weaksim.py`](run_ask_weaksim.py)
- [`analyze_weaksim.py`](analyze_weaksim.py)
- [`run_ask_qwen35_own.py`](run_ask_qwen35_own.py)
- [`analyze_ask35_control.py`](analyze_ask35_control.py)
- [`analyze_ask35_main.py`](analyze_ask35_main.py)
- [`plot_part1_figure.py`](plot_part1_figure.py)

Result files the index names:

- `part1_2026-10-12_002948.json` (not copied: result left out to keep the release under 120 MB)
- [`part2_gates_2026-10-12_003816.json`](results/part2_gates_2026-10-12_003816.json)
- [`part2_gates_2026-10-12_012613.json`](results/part2_gates_2026-10-12_012613.json)
- [`weaksim_2026-10-12_010042.json`](results/weaksim_2026-10-12_010042.json)
- [`weaksim_length_resid_2026-10-12_011408.json`](results/weaksim_length_resid_2026-10-12_011408.json)
- [`ask35_control_verdict_2026-10-12_123102.json`](results/ask35_control_verdict_2026-10-12_123102.json)
- `part1_figure_2026-10-12_120018.png` (not copied: figure (not copied))

## Files here

26 code and configuration files and 5 result files, copied byte for byte from `experiments/2026-10-11_is_this_a_test/` at commit `2feef331f3dacf7082a17d71a15b10a05fb6a3b6` of our internal repository. 70 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
