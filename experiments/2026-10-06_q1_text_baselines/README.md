# 2026-10-06_q1_text_baselines

2026-10-06 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

How well do text-only methods (a term-frequency text classifier, TF-IDF, and the LinuxArena paper's pairwise language-model judge) tell real from simulated LinuxArena attack runs, on the probe's tasks and folds?

## What we did

Phase 1 baselines on GPT-5.6-Sol runs: TF-IDF classifiers on the same tool outputs, folds and 32k-token cut as the Qwen3-32B probe, over a text cleanup ladder (from no cleanup up to all seven cumulative cleanups: times and dates, numbers and IDs, whitespace, harness format, diff lines, empty diff lines, editor output format) and held-out environments, plus GPT-5.6-Sol as a pairwise judge, with real-versus-real controls.

## What we found

TF-IDF 0.977 per run with no text cleanup vs 0.502 between two real runs (times masked). It falls to 0.667 with all seven cleanups. The pairwise judge names the real run 0.633 of the time. The honest real-versus-real 0.553 was a batch mix, 0.502 on same-batch pairs.

## In the post?

Not in the post.

## Status

Current (Phase 1 record)

## Builds on

[2026-10-06_data_checks](../2026-10-06_data_checks/) (splits v3)

## Main scripts

- [`run_tfidf_modal.py`](run_tfidf_modal.py)
- [`analyze_tfidf.py`](analyze_tfidf.py)
- [`analyze_ladder_table.py`](analyze_ladder_table.py)
- [`run_judge_repro.py`](run_judge_repro.py)
- [`analyze_paper_judge.py`](analyze_paper_judge.py)
- [`analyze_honest_floor_batch.py`](analyze_honest_floor_batch.py)
- [`run_stage_cohort.py`](run_stage_cohort.py)

Result files the index names:

- [`tfidf_summary_2026-10-06_134708.json`](results/tfidf_summary_2026-10-06_134708.json)
- [`tfidf_cleaned_summary_2026-10-06_171905.json`](results/tfidf_cleaned_summary_2026-10-06_171905.json)
- `tfidf_ladder_table_2026-10-06_170749.md` (not copied: lines over 300 characters)
- `paper_judge_summary_2026-10-06_130221.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- `judge_repro_2026-10-06_134502.json` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))
- [`honest_floor_by_batch_2026-10-07_210800.md`](results/honest_floor_by_batch_2026-10-07_210800.md)

## Files here

86 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-06_q1_text_baselines/` at commit `3d83154e8d252acf7177d53648f9eb34e9163a3a` of our internal repository. 232 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
