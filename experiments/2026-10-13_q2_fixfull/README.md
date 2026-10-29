# 2026-10-13_q2_fixfull

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

How much of the frozen tool-output detectors' simulated-vs-real gap does each post hoc simulator fix close, alone and together?

## What we did

Applied each of nine formatting fixes (insert review note, trailing newlines, line-number gutter, dpkg, clock, error text, view newline) and combinations to the 939 simulated attack runs (and 437 simulated honest runs) of GPT-5.6-Sol's cohort, re-extracted Qwen3-32B (layers 48, 63) and Seed-OSS-36B activations, and rescored with frozen probes; later amendments added two tab fixes and matched-false-alarm checks. Modal GPU.

## What we found

README (as-run JSONs at f9f2ea34): Qwen3-32B layer 63 gap at 1% is +10.36 as is; the full fix set (eight fixes) raises simulated catch by +11.93 points [+10.14, +13.67], reducing the second-real-minus-simulated gap from +10.36 to −1.57; str_replace trailing newline alone closes 54%. Seed-OSS-36B gap +13.63 falls to +5.17 under the full set (62% closed). The full set raises Qwen3-32B layer-63 false alarms on simulated honest runs by +0.74 points at 1%.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-09_fixed_sim](../2026-10-09_fixed_sim/) (fix rules); 2026-10-12 repaired arm of all 939 cells; [2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/); [2026-10-13_claimB_tab_census](../2026-10-13_claimB_tab_census/) (the two tab fixes)

## Main scripts

- [`stage_fixfull.py`](stage_fixfull.py)
- [`run_extract_qwen3_fx.py`](run_extract_qwen3_fx.py)
- [`g3_score_fx.py`](g3_score_fx.py)
- [`g3_score_f9.py`](g3_score_f9.py)
- [`g3_score_f10.py`](g3_score_f10.py)
- [`g3_score_f11.py`](g3_score_f11.py)
- [`analyze_fx_qwen3.py`](analyze_fx_qwen3.py)
- [`analyze_fx_seed.py`](analyze_fx_seed.py)
- [`analyze_matched_fa_fx.py`](analyze_matched_fa_fx.py)
- [`analyze_fx_mech_qwen3.py`](analyze_fx_mech_qwen3.py)
- [`analyze_f10f11_qwen3.py`](analyze_f10f11_qwen3.py)
- [`analyze_f10f11_seed.py`](analyze_f10f11_seed.py)

Result files the index names:

- [`seed_honest_fa_POSTHOC_2026-10-13_163345.json`](results/seed_honest_fa_POSTHOC_2026-10-13_163345.json)

## Files here

44 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_q2_fixfull/` at commit `9029fc409424cf82da706df37ac0850d77b49882` of our internal repository. 37 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
