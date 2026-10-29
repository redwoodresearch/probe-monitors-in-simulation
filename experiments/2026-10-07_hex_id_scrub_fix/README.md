# 2026-10-07_hex_id_scrub_fix

2026-10-07 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Did a bug in the text-cleanup ID rule (underscore-joined hex IDs were chopped instead of masked) leave a real-versus-simulated tell that inflated the text classifier's results?

## What we did

Counted the outputs the fix changes in each arm and refit the TF-IDF text classifier with times and dates masked, with numbers and IDs also masked, and with all seven cumulative cleanups (times and dates, numbers and IDs, whitespace, harness format, diff lines, empty diff lines, editor output format), before vs after the fix, with predictions written first.

## What we found

README verdict: the bug was real but did not matter; the fix changes about 0.2% of outputs per arm; AUROC with all seven cleanups 0.555 vs 0.556 before; real-versus-real 0.496 vs 0.495.

## In the post?

Not in the post.

## Status

Current (on a side branch)

## Builds on

[2026-10-06_q1_text_baselines](../2026-10-06_q1_text_baselines/)

## Main scripts

- [`analyze_mangling_counts.py`](analyze_mangling_counts.py)
- [`analyze_compare.py`](analyze_compare.py)

Result files the index names:

- [`mangling_counts_2026-10-07_131947.json`](results/mangling_counts_2026-10-07_131947.json)
- [`compare_2026-10-07_133339.json`](results/compare_2026-10-07_133339.json)

## Files here

4 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-07_hex_id_scrub_fix/` at commit `7a56ef01a5201d181fe744975b6d42caa737422d` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
