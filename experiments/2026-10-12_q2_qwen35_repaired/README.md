# 2026-10-12_q2_qwen35_repaired

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Do the two editor-output formatting fixes (rendering simulated text-editor insert and str_replace outputs the way the real tool does) change how often the attack detector read by Qwen3.5-27B catches simulated runs, filling the "after both fixes" cell for Qwen3.5-27B?

## What we did

Post hoc: extracted Qwen3.5-27B layer-63 activations for the 823 repaired simulated runs and 20 control runs on Modal (\$7.85), scored them with the frozen detector from the Qwen3.5-27B replication ([2026-10-12_q2_qwen35_reply_readout](../2026-10-12_q2_qwen35_reply_readout/)), and applied the pre-set decision rule of the original formatting-fix test on Qwen3-32B.

## What we found

Result JSON: at 5% repaired minus second real -1.52 points [-3.97, +1.37] vs -2.72 as simulated; reading "doesn't explain it" (the as-simulated gap was already small). No committed label changes under the gap-grew rule.

## In the post?

Not in the post.

## Status

Exploratory (post hoc)

## Builds on

[2026-10-12_q2_qwen35_reply_readout](../2026-10-12_q2_qwen35_reply_readout/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`run_score_q35fix.py`](run_score_q35fix.py)
- [`analyze_edit_effect_q35.py`](analyze_edit_effect_q35.py)
- [`check_gap_grew_label.py`](check_gap_grew_label.py)

Result files the index names:

- `edit_effect_q35_2026-10-13_003512.json` (not copied: result left out to keep the release under 120 MB)
- [`gap_grew_check_2026-10-13_013622.json`](results/gap_grew_check_2026-10-13_013622.json)
- [`q35fix_scores_repaired_2026-10-13_003323.json`](results/q35fix_scores_repaired_2026-10-13_003323.json)

## Files here

6 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-12_q2_qwen35_repaired/` at commit `8274ff69c93479b3530071fca9c9721e8e7e01ac` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
