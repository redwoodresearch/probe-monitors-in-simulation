# 2026-10-20_routec_simpool_numbers

2026-10-20 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

Recompute the numbers that depend on the final simulated honest pool, and give the attack-side Sol share a committed source.

## What we did

Recomputed, on the final pool (adding the 20 pilot runs): the share of simulated honest runs with Sol fallback, the failed simulated honest rate, Table 7 row 4's cohort and row 5's control; and reran the attack-side Sol share from committed records; Modal CPU.

## What we found

The post: Sol wrote part of the output in about 42% of simulated attack runs and about 6% of simulated honest runs; failed simulated honest runs 4.3% (2.9% without rate-limit failures).

## In the post?

In the post: Footnote on the simulator (about 42% attack, about 6% honest); footnote on failed honest runs (4.3%, 2.9%); Table 7 rows 4-5

## Status

Current

## Builds on

[2026-10-16_honest_sim_detector_all](../2026-10-16_honest_sim_detector_all/); [2026-10-16_attack_refit3](../2026-10-16_attack_refit3/)

## Main scripts

- [`run_routec_simpool_numbers.py`](run_routec_simpool_numbers.py)
- [`run_attack_sol_share.py`](run_attack_sol_share.py)

Result files the index names:

- `routec_simpool_numbers_2026-10-20_020501.json` (not copied: strings over 300 characters)
- `attack_sol_share_2026-10-20_023418.json` (not copied: strings over 300 characters)

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. For the 2 result files the index names, they match in 3 script-and-commit pairs, and in 1 the script was not in the folder at that commit.

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-20_routec_simpool_numbers/` at commit `36d8240fc714a2d41ab7f63c77abe10a76224156` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
