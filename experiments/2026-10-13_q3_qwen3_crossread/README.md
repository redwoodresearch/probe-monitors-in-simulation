# 2026-10-13_q3_qwen3_crossread

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Does a different reader, Qwen3-32B, reading Qwen3.5-27B's own acting runs, learn the 'simulator formatting means attack' shortcut that a mixed-recipe tool-output detector (trained on simulated attacks vs real honest runs) could learn?

## What we did

Extracted Qwen3-32B activations on Qwen3.5-27B's 196-task runs (main, second real run, planted and repaired arms; \$14.93), then trained real-trained and mixed cells at layers 39, 51, 63 (tool-output last token) on 183 tasks, with planted shifts, a shuffle null and a yardstick. Several monitor steps were held at a permission check.

## What we found

README (xread_mixed_2026-10-13_153151.json on S3): the mixed cell catches 38.3% of simulated attacks but 9.3% of second real runs at 1%, simulated-minus-real gap +29.0 [+22.4, +57.6] (+54.1 at 5%) against a yardstick of +2.2; the real-trained cell's gap is -3.3; reading '3: inconclusive' because the plant check failed. Coverage gate passed; extraction of 770 + 192 + 148 + 138 runs.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

The clean attack-detector test's cache and 196-task folds ([2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/))

## Main scripts

- [`run_extract_crossread.py`](run_extract_crossread.py)
- [`run_mixed_crossread.py`](run_mixed_crossread.py)
- [`run_claimc_crossread.py`](run_claimc_crossread.py)
- [`analyze_xread_mixed.py`](analyze_xread_mixed.py)
- [`analyze_xread_claimc.py`](analyze_xread_claimc.py)
- [`analyze_xread_monitor1.py`](analyze_xread_monitor1.py)

Result files the index names:

- [`coverage_main_limit8_2026-10-13_012956.json`](results/coverage_main_limit8_2026-10-13_012956.json)
- `manifest_xread_mixed_s3_v1.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_scoring_s3_v1.json` (not copied: result left out to keep the release under 120 MB)
- `manifest_xread_packaging_s3_v1.json` (not copied: strings over 300 characters)

## Files here

20 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_q3_qwen3_crossread/` at commit `5eae6ea3a8d1e8ff81e2b01c270de454cd1dba88` of our internal repository. 23 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
