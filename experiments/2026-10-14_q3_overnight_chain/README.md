# 2026-10-14_q3_overnight_chain

2026-10-14 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Turn the scale-up launches into probe-ready data: relay each finished launch's files, stage them in the reader format, extract Qwen3.5-27B activations, and label attempts, with every step gated.

## What we did

chain.py ran per batch (the main and simulated follow-up launches for the quarters of the +311 added tasks; later extended to the splits of the tasks with new main tasks, interleaved arms, the 493 paper cells, epoch 2 and full-length re-stages): box-side file hashing, staging manifests per world (the two real honest arms, run 1, run 2, simulated), H200 extraction of every layer with reproduction gates, attempt labelling, plus many provenance, routing and supersession records.

## What we found

The activation log lists the chain's extraction dumps : 242 extraction dumps across the scale-up strata (ACTIVATIONS_8a53f74ee.md lines 528–769), each made by stage_chain.py and extract_chain.py on H200 with a reproduction gate.

## In the post?

In the post: Appendix A runs and activations (the +311 added tasks and later task sets of the 1,313 tasks)

## Status

Current

## Builds on

[2026-10-14_q3_scaleup_311](../2026-10-14_q3_scaleup_311/); [2026-10-14_q3_new_mains_launch](../2026-10-14_q3_new_mains_launch/)

## Main scripts

- [`chain.py`](chain.py)
- [`box_files.py`](box_files.py)
- [`extract_chain.py`](extract_chain.py)
- [`labels_chain.py`](labels_chain.py)
- [`hygiene_chain.py`](hygiene_chain.py)
- [`compare_repro.py`](compare_repro.py)
- [`fmt_gate.py`](fmt_gate.py)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/378d4c7b5/stage_chain.py`](as_run/378d4c7b5/stage_chain.py) from commit `378d4c7b5`: Staged the runs whose activations the post's probes read; release/pipeline/units.py cites its line numbers at this commit. The copy in this folder's snapshot is a newer version of the script.
- [`as_run/378d4c7b5/stage_q4b.py`](as_run/378d4c7b5/stage_q4b.py) from commit `378d4c7b5`: Staged the runs whose activations the post's probes read (the system-prompt rule for attack runs and what is masked); release/pipeline/units.py cites its line numbers at this commit. The copy in this folder's snapshot is a newer version of the script.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. Other result files, from earlier or side runs, give 1383 script-and-commit pairs that match, 1240 that differ and 2 where the script was not in the folder; we did not copy those versions. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

879 code and configuration files and 611 result files, copied byte for byte from `experiments/2026-10-14_q3_overnight_chain/` at commit `59de09da98f51e2ea9e9a5fc3d8c5ed7e1677c14` of our internal repository. 4073 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 2 files under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
