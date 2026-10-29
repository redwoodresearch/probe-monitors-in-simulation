# 2026-10-14_q2_ffm_iter1

2026-10-14 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

After the earlier script fixes to the simulator's output formatting used here (the first nine and the eleventh; the tenth was omitted), which kind of tool-output content still differs most between real and simulated honest runs, can a script fix it, and does that fix change a text classifier's ability to separate them and the probes' simulated-minus-real catch gap?

## What we did

Counted classes of content in real and simulated honest tool outputs (GPT-5.6-Sol's runs, 437 tasks), found Windows-style line endings (CRLF) in curl header output, and wrote a twelfth fix that adds them; it re-extracted Qwen3-32B activations (the other four reader models got a tokenizer-only check that the fix reaches their tokens), then measured a text classifier, the gap of the earlier phase's headline Qwen3-32B probe and a refit of the mixed probe (trained on simulated attacks against real honest runs), with dose and per-model checks.

## What we found

README: CRLF in curl output in 29.3% of real vs 1.6% of simulated honest runs; the curl-header line-ending repair (the twelfth fix) lowered the text classifier's AUROC from 0.6133 to 0.5883; the gap of the previously selected main probe configurations barely moved (−0.08 points, 95% −0.21 to +0.08); recipe (c) (simulated-attack versus real-honest training) equivalent to zero within ±1 point; a post hoc finding that the tab-formatting repair (the eleventh fix) cut recipe (c)'s gap by about two thirds.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

The 2026-10-13 full-fix and tab-fix stagings

## Main scripts

- [`stage_f12.py`](stage_f12.py)
- [`classify_f12.py`](classify_f12.py)
- [`score_gap.py`](score_gap.py)
- [`cell_c.py`](cell_c.py)
- [`reach_f12.py`](reach_f12.py)
- [`dose_count.py`](dose_count.py)
- [`model_split.py`](model_split.py)
- [`run_extract_qwen3_f12.py`](run_extract_qwen3_f12.py)

Result files the index names:

- [`counts_2026-10-14_094727.json`](results/counts_2026-10-14_094727.json)
- [`stage_f12_2026-10-14_101631.json`](results/stage_f12_2026-10-14_101631.json)
- [`classify_f12_POSTHOC_2026-10-14_101949.json`](results/classify_f12_POSTHOC_2026-10-14_101949.json)
- `gap_2026-10-14_105333.json` (not copied: result left out to keep the release under 120 MB)
- `cell_c_2026-10-14_110858.json` (not copied: result left out to keep the release under 120 MB)
- `dose_count_2026-10-14_110529.json` (not copied: result left out to keep the release under 120 MB)

## Files here

19 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-14_q2_ffm_iter1/` at commit `c7da8143ecba9860f538ff823fdd8d5758037653` of our internal repository. 23 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
