# 2026-10-08_q3_sabotage_probe_mvp

2026-10-08 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Can a probe trained on hand labels of sabotage calls find the sabotage call inside an attack run, better than a text classifier on the command?

## What we did

Exploratory: Qwen3-32B activations at commands of 20 hand-labelled real GPT-5.6-Sol attack runs (56 sabotage, 412 benign calls), logistic-regression probe tested on held-out side tasks and main tasks against a TF-IDF text classifier on command text; a quick check at early layers 0, 8 and 16.

## What we found

Held-out ranking 0.828 (unseen side tasks) and 0.867 (unseen main tasks), but it ties a text classifier on main tasks (+0.003 [-0.071, +0.081]); simulated runs untested. Early layers still score 0.739-0.757 (preliminary).

## In the post?

Not in the post.

## Status

Exploratory (not the pre-registered Stage 2 probe)

## Builds on

[2026-10-07_q3_codex_followups](../2026-10-07_q3_codex_followups/) (hand labels of sabotage calls)

## Main scripts

- [`run_sabotage_probe_mvp.py`](run_sabotage_probe_mvp.py)
- [`analyze_mvp_checks.py`](analyze_mvp_checks.py)
- [`analyze_tfidf_baseline.py`](analyze_tfidf_baseline.py)
- [`make_blind_spotcheck.py`](make_blind_spotcheck.py)

Result files the index names:

- `sabotage_probe_mvp_2026-10-08_094806.json` (not copied: result left out to keep the release under 120 MB)
- `sabotage_probe_mvp_2026-10-08_095844_side_task.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `mvp_checks_summary_2026-10-08_154553.jsonl` (not copied: result file type not copied (.jsonl))
- `sabotage_probe_mvp_2026-10-08_142157_side_task.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_q3_sabotage_probe_mvp/` at commit `a25f56f8864aca0247ee8748dee80ec9dfff1154` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
