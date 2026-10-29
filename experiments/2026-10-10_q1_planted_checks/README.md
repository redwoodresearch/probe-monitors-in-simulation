# 2026-10-10_q1_planted_checks

2026-10-10 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does the Phase 1 real-versus-simulated probe respond to the meaning 'this is a simulation', to planted years, or to the simulator's own error lines?

## What we did

Planted an explicit simulation line, planted years and a neutral-line control in GPT-5.6-Sol runs, refit and scored Qwen3-32B probes; amendments added a meaning control with the frozen ladder probe at all seven cumulative cleanups (times and dates, numbers and IDs, whitespace, harness format, diff lines, empty diff lines, editor output format), a screen of invented 'Invalid view_range' errors and a matched planted error-line check.

## What we found

The explicit line is detected (0.996-1.000) but so is a neutral line, so the check shows the probe detects an inserted line; it cannot show whether the line's meaning matters (a later PLAN.md notes that ceiling-level detection cannot establish this). A planted simulator error line moves run scores +0.105 early and +0.098 mid toward simulated (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary)

## Builds on

[2026-10-06_q1_segment_end](../2026-10-06_q1_segment_end/); [2026-10-06_all_layers_extraction](../2026-10-06_all_layers_extraction/)

## Main scripts

- [`run_stage_plants.py`](run_stage_plants.py)
- [`run_pack_plants.py`](run_pack_plants.py)
- [`run_grid_plants.py`](run_grid_plants.py)
- [`spawn_grid_plants.py`](spawn_grid_plants.py)
- [`analyze_plants.py`](analyze_plants.py)
- [`analyze_meaning.py`](analyze_meaning.py)
- [`analyze_screen.py`](analyze_screen.py)
- [`score_error.py`](score_error.py)
- [`analyze_error.py`](analyze_error.py)

Result files the index names:

- `q1plants_grid_2026-10-10_213943.json` (not in this folder at the source commit)
- `q1plants_verdicts_2026-10-10_213956.json` (not in this folder at the source commit)
- `error_scores_2026-10-10_234626.json` (not in this folder at the source commit)
- `error_verdicts_2026-10-10_234627.json` (not copied: result left out to keep the release under 120 MB)

## Files here

13 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-10_q1_planted_checks/` at commit `8c342502756fe286a5a55e09f4f940a92fb18fe4` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
