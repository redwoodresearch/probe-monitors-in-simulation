# 2026-10-13_q3_recompute_kit

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Can the project lead recompute the four headline numbers of the simulation-trained-probe work himself, and read five transcript pairs drawn at random?

## What we did

Four short standard-library scripts that read each headline result by key, check its sha256 and print the number with its interval (how far the mixed-recipe probe overstates catch, the values from the test of why repaired probes still overstate, simulation-trained vs count-matched real training, and the simulated-minus-real grading gaps), plus five seeded transcript pairs listed by ID and S3 path only. \$0.

## What we found

No results of its own; HOWTO_recompute.md lists the claims it checks (trap 52 to 79 points across readers; count-matched Qwen3-32B +7.6, Seed +2.1; grading gaps +2.3 and 0.0).

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/); [2026-10-13_why_steps](../2026-10-13_why_steps/); [2026-10-13_q3_trap_figure](../2026-10-13_q3_trap_figure/)

## Main scripts

- [`cell_i_trap.py`](cell_i_trap.py)
- [`cell_ii_why_steps.py`](cell_ii_why_steps.py)
- [`cell_iii_count_matched.py`](cell_iii_count_matched.py)
- [`cell_iv_grading_gaps.py`](cell_iv_grading_gaps.py)
- [`make_pairs.py`](make_pairs.py)

Result files the index names:

- `pairs_seed20261013.csv` (not copied: file type not copied (.csv))

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_recompute_kit/` at commit `35c1e4969d85f79126e6f21949a65480286bfb21` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
