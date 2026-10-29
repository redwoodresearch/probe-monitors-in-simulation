# 2026-10-12_q2_fix939_checks

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Re-rendering all 939 simulated runs' editor outputs in the real tool's format closed the first-phase attack detector's real-versus-simulated gap at tool-output readouts: does that hold outside the 100 pilot cells where it was found, and is it an artifact of where the readout lands?

## What we did

About \$0.04 on Modal: reran the re-rendering analysis on the 839 non-pilot cells, counted which token the readout lands on before and after the edit, re-pooled the detector's frozen layer-63 scores without readouts on chat-template tokens, and added swap nulls.

## What we found

README: outside the pilot the edit still raises catch by +12.9 points [+10.5, +14.7] at 5%, edited minus second real -1.0 [-4.4, +1.9]; for inserts the readout moves onto the review-note template (1.1% to 91.4% of simulated outputs), so a readout-position artifact is plausible.

## In the post?

Not in the post.

## Status

Current (preliminary; on a side branch)

## Builds on

[2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/); [2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/)

## Main scripts

- [`run_coverage.py`](run_coverage.py)
- [`run_readout_tokens.py`](run_readout_tokens.py)
- [`run_repool.py`](run_repool.py)
- [`analyze_fix939_followups.py`](analyze_fix939_followups.py)
- [`analyze_repool.py`](analyze_repool.py)
- [`analyze_repool_swap_null_POSTHOC.py`](analyze_repool_swap_null_POSTHOC.py)

Result files the index names:

- `fix939_followups_2026-10-12_015629.json` (not in this folder at the source commit)
- `readout_tokens_2026-10-12_020248.json` (not in this folder at the source commit)
- `repool_reading_2026-10-12_021657.json` (not in this folder at the source commit)

## Files here

12 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-12_q2_fix939_checks/` at commit `6b3758a187cf250d73077ffb45f2ff38a08b2a8d` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
