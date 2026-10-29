# 2026-10-12_q2_meanpool_36

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does the in-domain tool-output attack detector (trained on real attack vs real honest runs) still catch simulated attack runs less often than real ones when each tool output is read by averaging its content tokens instead of taking its last token?

## What we did

Extracted mean-pooled Qwen3-32B activations on Modal for the 939 paired task cells (simulated runs as simulated and with the editor-output formatting repairs), refit the detector recipe with the mean-pooled readout, and compared it with the last-token detector on the same cells and bootstrap draws. Billed \$95.96.

## What we found

README: mean-pooled simulated minus second real at 5% +1.28 points [-1.12, +2.86] ("inconclusive": the +10 plant could not be built) vs -14.27 for last-token; paired readout difference +15.55 [+11.48, +18.68]. Review reproduced the last-token side exactly.

## In the post?

Not in the post.

## Status

Exploratory (preliminary)

## Builds on

[2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`run_score_fix_meanpool.py`](run_score_fix_meanpool.py)
- [`analyze_meanpool.py`](analyze_meanpool.py)
- [`check_lasttoken_inputs.py`](check_lasttoken_inputs.py)

Result files the index names:

- `meanpool_reading_2026-10-12_160554.json` (not in this folder at the source commit)
- [`fix_scores_meanpool_2026-10-12_160432.json`](results/fix_scores_meanpool_2026-10-12_160432.json)

## Files here

5 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-12_q2_meanpool_36/` at commit `bc683d4a4cbd230497469507d8c3667cd3e9a6fc` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
