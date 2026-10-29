# 2026-10-13_claimB_read10

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

What remains different between fully fixed simulated honest runs and real honest runs: simulator content, the agent's behaviour, or run-to-run consistency?

## What we did

Drew 10 of the top-10%-scoring fully fixed simulated honest runs (scored by the Qwen3-32B mixed probe trained on fixed simulated attacks vs real honest runs) with a stated seed and read each beside a real honest run of the same task, classing every observed difference; script checks ran on Modal CPU. GPT-5.6-Sol runs.

## What we found

counts_read10_2026-10-13.json (LLM reading, partly script-checked): over the 10 random runs, 14 content, 13 behaviour, 2 consistency, 1 unclear and 10 not-a-difference observations, plus 21 tags matching existing repair families; 12 first-flagged tool-set differences were reclassified as not a difference (simulator's tools).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (run scores of the mixed fix-full probe); [2026-10-13_q2_fixfull](../2026-10-13_q2_fixfull/) (fixed simulated honest arm)

## Main scripts

- [`draw_read10.py`](draw_read10.py)
- [`movers_read10.py`](movers_read10.py)
- [`checks_read10.py`](checks_read10.py)
- [`run_checks_modal.py`](run_checks_modal.py)

Result files the index names:

- [`draw_2026-10-13_170032.json`](results/draw_2026-10-13_170032.json)
- `movers_2026-10-13_170235.json` (not copied: result left out to keep the release under 120 MB)
- `checks_2026-10-13_172945.json` (not copied: result left out to keep the release under 120 MB)
- `counts_read10_2026-10-13.json` (not copied: strings over 300 characters)

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_claimB_read10/` at commit `22ad0f386d40406de6de80c4db276dae4c828fab` of our internal repository. 5 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
