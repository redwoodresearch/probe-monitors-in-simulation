# 2026-10-14_q3_onpol_learning_curve

2026-10-14 · part of [Side questions about the main comparison](../../docs/experiments.md#side-questions-about-the-main-comparison)

## Question

Does adding k of Qwen3.5-27B's own attack runs (real, or repaired simulated) and its own honest runs to the training data of the earlier phase's headline probes (trained on GPT-5.6-Sol's runs) close the gap to a probe trained on Qwen3.5-27B's own runs, as it did when Qwen3.5-27B itself was the reader?

## What we did

Refit layer-48 probes averaged over tool outputs, read by Qwen3-32B and Seed-OSS-36B, on Sol's runs plus k = 10, 30 or 100 of Qwen3.5-27B's own attack runs (real or repaired simulated, with and without its own honest runs), and measured catch of the second real attack run on 183 tasks.

## What we found

Qwen3-32B reader at 5%: catch rose from 47.5 (k = 0) to 80.8 with 100 own real attacks, closing 76% (95% 59–89%) of the gap to the on-policy 91.3; simulated own attacks did as well (S − R at k100 +3.8, 95% −1.8 to +7.2); without own honest runs catch fell (lc_2026-10-14_032758.printout.txt).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-14_q3_onpolicy_titlecells](../2026-10-14_q3_onpolicy_titlecells/) (packs)

## Main scripts

- [`run_lc.py`](run_lc.py)
- [`report_lc.py`](report_lc.py)
- [`post_lc_baseb.py`](post_lc_baseb.py)

Result files the index names:

- [`lc_baseb_share_2026-10-14_034114.json`](results/lc_baseb_share_2026-10-14_034114.json)

## Files here

7 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-14_q3_onpol_learning_curve/` at commit `ad9a70877306ac4fc28834263822bf4b047334aa` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
