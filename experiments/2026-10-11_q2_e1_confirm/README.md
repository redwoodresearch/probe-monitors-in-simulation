# 2026-10-11_q2_e1_confirm

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Hide-or-mention test, fresh confirmation: in new runs, does the frozen deception probe read higher on change statements the agent was told to conceal than on ones it was told to mention?

## What we did

Generated fresh hide and mention reports from new Qwen3.5-27B honest-run contexts (correctly resolved) on Modal, selected change-statement sentences by script, and applied the pre-registered rule, margin, gates and a planted shift.

## What we found

Yes under its rule, +0.58 standard deviations (0.20 to 0.95) over 39 runs, and the planted shift was found; without the file path's words +0.38 (-0.04 to 0.78), so the path's words may carry part of it.

## In the post?

Not in the post.

## Status

Current (the hide-or-mention test's result of record; preliminary)

## Builds on

[2026-10-11_q2_e1_sentences](../2026-10-11_q2_e1_sentences/); [2026-10-09_q2_deception_apollo](../2026-10-09_q2_deception_apollo/)

## Main scripts

- [`select_confirm.py`](select_confirm.py)
- [`build_confirm_index.py`](build_confirm_index.py)
- [`run_confirm.py`](run_confirm.py)
- [`analyze_confirm.py`](analyze_confirm.py)
- [`power_check.py`](power_check.py)
- [`masks_lib.py`](masks_lib.py)
- [`sensitivity_POSTHOC_floor.py`](sensitivity_POSTHOC_floor.py)

Result files the index names:

- `confirm_reading_2026-10-11_185042.json` (not copied: result left out to keep the release under 120 MB)
- [`confirm_masks_2026-10-11_185020.json`](results/confirm_masks_2026-10-11_185020.json)
- [`power_check_2026-10-11_164216.json`](results/power_check_2026-10-11_164216.json)
- [`floor_sensitivity_POSTHOC_2026-10-11_191531.json`](results/floor_sensitivity_POSTHOC_2026-10-11_191531.json)

## Files here

8 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-11_q2_e1_confirm/` at commit `0a9c4ae0c8abd51508402bb1cd763f1dad008924` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
