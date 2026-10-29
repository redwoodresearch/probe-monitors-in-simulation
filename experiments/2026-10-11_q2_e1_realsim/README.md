# 2026-10-11_q2_e1_realsim

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Does the hide-or-mention deception-probe response look the same on GPT-5.6-Sol's real and GPT-6-Astra-simulated honest runs, within plus or minus 0.5 honest SD, as on two real runs?

## What we did

Two pilots on Modal: first with tools available at the report turn, then a redesign with no tools offered; post hoc projection of options.

## What we found

README: closed; the first pilot gave 2, 1 and 2 usable pairs per arm (Qwen made tool calls in 278 of 306 report turns); the redesign gave 66% usable reports against a 70% bar and a projected chance of a clear answer of 0.0.

## In the post?

Not in the post.

## Status

Void (closed after pilots)

## Builds on

[2026-10-11_q2_e1_sentences](../2026-10-11_q2_e1_sentences/) (PLAN_REAL_VS_SIM.md); [2026-10-11_q2_e1_confirm](../2026-10-11_q2_e1_confirm/)

## Main scripts

- [`build_realsim_index.py`](build_realsim_index.py)
- [`run_realsim.py`](run_realsim.py)
- [`run_realsim2.py`](run_realsim2.py)
- [`analyze_realsim_pilot.py`](analyze_realsim_pilot.py)
- [`analyze_realsim2_pilot.py`](analyze_realsim2_pilot.py)
- [`project_POSTHOC_options.py`](project_POSTHOC_options.py)
- [`count_replay_errors_POSTHOC.py`](count_replay_errors_POSTHOC.py)

Result files the index names:

- `pilot_readout_2026-10-11_171725.json` (not copied: result left out to keep the release under 120 MB)
- [`redesign_pilot_readout_2026-10-11_182405.json`](results/redesign_pilot_readout_2026-10-11_182405.json)
- [`options_projection_POSTHOC_2026-10-11_191535.json`](results/options_projection_POSTHOC_2026-10-11_191535.json)
- [`replay_errors_POSTHOC_2026-10-12_065012.json`](results/replay_errors_POSTHOC_2026-10-12_065012.json)

## Files here

9 code and configuration files and 5 result files, copied byte for byte from `experiments/2026-10-11_q2_e1_realsim/` at commit `471936c47aa0a7b4d04f028b1eadb891fb8e670d` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
