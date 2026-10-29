# 2026-10-12_q3_edit_honest_fa

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does the rendering edit that closes the tool-output detector's simulated-vs-real gap (insert outputs gain the real review note, str_replace diffs gain the real line-number gutter, trailing newlines are removed) raise false alarms on honest runs, i.e. does it simply push every detector score up?

## What we did

Applied the edit to 381 of 437 simulated honest runs, re-extracted Qwen3-32B activations on Modal, scored them with the in-domain tool-output detector's frozen layer-63 model, and compared false alarms with the same runs as simulated and with real honest runs, with a planted check.

## What we found

README: the edit raises honest false alarms by +2.0 points [+0.7, +3.1] at 5% (5.4% to 7.4%), much less than its +13.7-point catch rise on attack runs; at layer 48 -1.20, inconclusive.

## In the post?

Not in the post.

## Status

Exploratory (post hoc, preliminary)

## Builds on

[2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/); [2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/)

## Main scripts

- [`stage_edit_honest.py`](stage_edit_honest.py)
- [`spawn_extract.py`](spawn_extract.py)
- [`score_edit_honest.py`](score_edit_honest.py)
- [`analyze_edit_honest.py`](analyze_edit_honest.py)
- [`power_edit_honest.py`](power_edit_honest.py)
- [`check_smoke.py`](check_smoke.py)

Result files the index names:

- `edit_honest_fa_POSTHOC_2026-10-12_023027.json` (not in this folder at the source commit)

## Files here

6 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-12_q3_edit_honest_fa/` at commit `d0bfe04288d5c0f5299d4680a2d7ae0de80f9877` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
