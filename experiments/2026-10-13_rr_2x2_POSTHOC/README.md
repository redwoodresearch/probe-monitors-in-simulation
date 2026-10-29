# 2026-10-13_rr_2x2_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

On Qwen3.5-27B's own runs with its reasoning removed, does a mixed-trained probe show the 'simulated = attack' shortcut at the reply or command readout?

## What we did

2x2 of real-trained vs mixed cells at the tool-output and command readouts on the reasoning-stripped own-run cache, whole-run and first-j-commands versions (172 to 165 tasks), with a real-vs-real yardstick; plus an inventory showing no on-policy simulated honest runs exist beyond 20 tasks. Modal CPU.

## What we found

README: at the pre-registered first 3 commands, the mixed probe at the reply readout catches simulated attacks +12.0 [+3.7, +19.9] points more than second real runs at 1% (paired vs real-trained +17.7), above the real-vs-real gap (-0.6); the shortcut uses 14.7% of the room, against 73 to 93% off-policy; the command readout and whole-run versions are inconclusive.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (stripped no-agent cache, records of the first-flag-position analysis)

## Main scripts

- [`run_rr_2x2.py`](run_rr_2x2.py)
- [`run_rr_early.py`](run_rr_early.py)
- [`count_simhonest_POSTHOC.py`](count_simhonest_POSTHOC.py)
- [`verify_rr_records_archive.py`](verify_rr_records_archive.py)

Result files the index names:

- `rr_2x2_POSTHOC_2026-10-13_182358.printout.txt` (not copied: lines over 300 characters)
- [`rr_early_POSTHOC_2026-10-13_182659.printout.txt`](results/rr_early_POSTHOC_2026-10-13_182659.printout.txt)
- `simhonest_inventory_2026-10-14_001113.jsonl` (not copied: result file type not copied (.jsonl))

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_rr_2x2_POSTHOC/` at commit `fd012160fd1b355cb64ba2edb2fb75f3baaafc3e` of our internal repository. 15 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
