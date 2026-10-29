# 2026-10-17_routec_slot_readout

2026-10-17 · part of [The main comparison](../../docs/experiments.md#the-main-comparison)

## Question

Read every number the write-up quotes from the final pooled records of the analysis on all 1,313 tasks, inside a Modal container rather than on the laptop.

## What we did

A Modal CPU reader that checks input hashes and builds the write-up's number strings from the final records (main results at 1% and 5%, the sweep of alarm levels, and strict alarm levels) and from the simulation detector at layer 48, with control runs on an earlier version of the main record built on a 171-run simulated honest pool.

## What we found

Slot files slots_routeC_writing_rc1a.json, slots_routeC_writing_rc1b.json, slots_routeC_writing_rc1b_detl48.json and slots_routeC_writing_rc2.json written.

## In the post?

In the post: Numbers throughout the post (Figure 1, Tables 5, 7, 8, Appendices C-H)

## Status

Current

## Builds on

[2026-10-15_fpr_sweep_507_prep](../2026-10-15_fpr_sweep_507_prep/); [2026-10-16_attack_refit3](../2026-10-16_attack_refit3/)

## Main scripts

- [`run_slot_readout.py`](run_slot_readout.py)
- [`slot_table.py`](slot_table.py)
- [`s2_gate.jq`](s2_gate.jq)

Result files the index names:

- `slots_routeC_writing_rc1a.json` (not in this folder at the source commit)
- `slots_routeC_writing_rc1b.json` (not copied: strings over 300 characters)
- `slots_routeC_writing_rc1b_detl48.json` (not copied: strings over 300 characters)
- `slots_routeC_writing_rc2.json` (not in this folder at the source commit)

## As-run versions

The copies above are the folder's latest versions. These versions, which differ from them, produced results the index names or the post uses (see each note):

- [`as_run/283a711ab/run_slot_readout.py`](as_run/283a711ab/run_slot_readout.py) from commit `283a711ab`: Wrote the statistics tables, the numbers the write-up reads from the pooled records. The script is byte-identical at commits 283a711ab and 6f86f2b0b, which both produced tables. The copy in this folder's snapshot is a newer version.

**As-run check.** Some result files here record the commit of the code they ran with. We compared the main scripts listed above at those commits with the copies here. Other result files, from earlier or side runs, give 3 script-and-commit pairs that match; we did not copy those versions. The differing versions behind results the index names or the post uses are under `as_run/` (above).

## Files here

5 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-17_routec_slot_readout/` at commit `6d77ea36908000cf282e5b52e16b288da058db2d` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text). The 1 file under `as_run/` come from the commits named above.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
