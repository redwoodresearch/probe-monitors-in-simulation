# 2026-10-13_q2_fresh_honest

2026-10-13 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do the frozen detectors' honest false-alarm rates hold on main tasks they have never seen?

## What we did

Ran GPT-5.6-Sol honest on 327 LinuxArena public main tasks no cohort had used (654 planned, 634 eligible on 317 tasks) in seven stages, extracted Qwen3-32B and Seed-OSS-36B activations, and scored six frozen detectors against their cohort false-alarm rates under a frozen rule, with an independent Codex recompute and a per-environment diagnostic. API spend \$316.31 plus Modal extraction.

## What we found

README (score_fresh_s1234567_2026-10-14_080417.json): at the 1% threshold no detector 'holds within 2 points'; false alarms are higher on fresh tasks for all six (for example the Qwen3-32B last-token tool-output attack detector at layer 63: 2.51% vs 1.09%, +1.42 [+0.27, +2.56]; the Seed-OSS-36B tool-output attack detector: 3.62% vs 1.55%); two read 'higher on fresh tasks', four 'inconclusive'; the prediction 'all six hold' was wrong. 10 tasks found broken (patch does not apply or placeholder task).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_g3_new_reader](../2026-10-13_g3_new_reader/) (new-reader detector); the earlier attack detector; [2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (recipe and real-trained cells)

## Main scripts

- [`make_fresh_tasks.py`](make_fresh_tasks.py)
- [`launch_fresh.sh`](launch_fresh.sh)
- [`chain_fresh.py`](chain_fresh.py)
- [`stage_fresh_modal.py`](stage_fresh_modal.py)
- [`run_extract_qwen3_fresh.py`](run_extract_qwen3_fresh.py)
- [`score_fresh_env.py`](score_fresh_env.py)
- [`compare_parity.py`](compare_parity.py)
- [`compare_parity_seed.py`](compare_parity_seed.py)
- [`make_readme.py`](make_readme.py)

Result files the index names:

- `score_fresh_s1234567_2026-10-14_080417.json` (not in this folder at the source commit)
- `score_fresh_env_s4to7_2026-10-14_080707.json` (not in this folder at the source commit)
- `s3_manifest_v1.json` (not copied: strings over 300 characters)
- [`errored_runs_2026-10-14.md`](results/errored_runs_2026-10-14.md)

## Files here

35 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_q2_fresh_honest/` at commit `e60c271f3e5651572cf1c0cf5d96a63c01740114` of our internal repository. 13 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
