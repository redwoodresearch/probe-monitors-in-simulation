# 2026-10-13_q2_new_task_pool

2026-10-13 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

How many new LinuxArena main tasks with a usable simulator reference remain for more simulated honest runs (outside the 437 used and the sealed confirmation set), and what would they cost?

## What we did

One Modal CPU function counted main tasks from committed records (references, scored runs, exclusions, the sealed set) and priced additional runs. \$0.

## What we found

task_pool_POSTHOC_2026-10-13_180047.json: 0 main tasks available with a usable reference among the fresh honest-reference rollouts outside the 437 and the sealed set; 105 'uncovered' mains (reference only among the original 200 real-honest reference runs, which overlap the compared cohort) not sealed.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-06_astra_sim_honest](../2026-10-06_astra_sim_honest/) (references); [2026-10-11_q2_new_real_runs](../2026-10-11_q2_new_real_runs/) (sealed set)

## Main scripts

- [`run_task_pool.py`](run_task_pool.py)

Result files the index names:

- `task_pool_POSTHOC_2026-10-13_180047.json` (not copied: result left out to keep the release under 120 MB)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q2_new_task_pool/` at commit `cea8646e215c50c75cc1ed6b8c37d52aa598c3fa` of our internal repository. 1 other file in that folder was not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
