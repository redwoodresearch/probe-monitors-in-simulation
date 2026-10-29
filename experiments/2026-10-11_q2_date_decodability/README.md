# 2026-10-11_q2_date_decodability

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Is the clock difference between two real runs (which a word counter detects at 0.690) decodable from Qwen3-32B's activations where it occurs, even though run-level probes miss it?

## What we did

Trained a probe aimed at the 5,170 tool outputs that follow a time or date (uncleaned text and with times masked), with a planted-year positive control and a run-label probe at weak regularization, on stored activations via Modal.

## What we found

README: partial; the aimed probe reaches 0.662 (null maximum 0.522) and drops to 0.499 with times masked; positive control 0.925; run-label probe 0.512. A targeted probe decodes the timestamp difference (0.66) that run-level probes miss (0.51).

## In the post?

Not in the post.

## Status

Current (thread closed)

## Builds on

[2026-10-06_segment_end_extraction](../2026-10-06_segment_end_extraction/) (Qwen3-32B activations); [2026-10-08_q2_date_histogram](../2026-10-08_q2_date_histogram/)

## Main scripts

- [`run_date_decode.py`](run_date_decode.py)
- [`analyze_date_decode.py`](analyze_date_decode.py)

Result files the index names:

- [`date_decode_2026-10-11_011130.json`](results/date_decode_2026-10-11_011130.json)

## Files here

2 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_q2_date_decodability/` at commit `a37eb435d078b048dae2ed78b7ba2900ee596135` of our internal repository. 3 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
