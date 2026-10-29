# 2026-10-12_q2_pinned_layers

2026-10-12 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

For the in-domain tool-output attack detector, is the simulated-vs-real catch gap caused by the readout (last token vs mean over each output's tokens) or by the layer (48 vs 63)?

## What we did

A \$0 two-by-two from stored scores, with the last-token and mean-pooled readouts each fixed at layer 48 and at layer 63 in every fold; a second addendum scored the last-token layer-48 detector on the simulated runs repaired with the editor-output formatting fixes.

## What we found

At 5%: last-token layer 48 +4.15 [+0.99, +7.21]; last-token layer 63 -14.27 [-17.69, -10.83]; mean-pooled layer 48 +1.28; layer 63 -2.82; repaired arm at layer 48 +0.80 [-2.60, +3.81], "explains it".

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_q2_meanpool_36](../2026-10-12_q2_meanpool_36/); [2026-10-06_q3_attack_detector](../2026-10-06_q3_attack_detector/); [2026-10-08_q3_swap_cue](../2026-10-08_q3_swap_cue/)

## Main scripts

- [`analyze_pinned.py`](analyze_pinned.py)
- [`run_score_fix_lt48.py`](run_score_fix_lt48.py)
- [`analyze_lt48_repaired.py`](analyze_lt48_repaired.py)

Result files the index names:

- `pinned_layers_2026-10-12_182642.json` (not in this folder at the source commit)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_q2_pinned_layers/` at commit `d9da21bfcb799384ac06bccfb27cdd662d0d03e4` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
