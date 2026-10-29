# 2026-10-15_q3_simhonest_311

2026-10-15 · part of [Building the attack and honest runs](../../docs/experiments.md#building-the-attack-and-honest-runs)

## Question

Collect simulated honest runs for the +311 added tasks (and later the tasks with new main tasks and the paper's tasks) so simulated thresholds can be set on them.

## What we did

One simulated-honest launch per quarter of the +311 added tasks (forked from each main's real honest run, same frozen simulator and harness), with configs for batches of the tasks with new main tasks and batches of the paper's remaining 493 tasks, retries and rate-limit reruns.

## What we found

Launch artifacts and counted-vs-pinned checks; scope for the +311 added tasks about \$6.5k raw (per the experiment's written rules); dumps logged in ACTIVATIONS.md (the simulated-honest dumps for the 311 added tasks and for the tasks with new main tasks).

## In the post?

In the post: indirect: simulated honest pool for the simulated threshold (Figure 1, Appendix A)

## Status

Current

## Builds on

2026-10-14 follow-up launches for the +311 added tasks

## Main scripts

- [`make_configs.py`](make_configs.py)
- [`simh311_box.sh`](simh311_box.sh)
- [`simhnm_box.sh`](simhnm_box.sh)
- [`simh_sp493_box.sh`](simh_sp493_box.sh)
- [`start_q.sh`](start_q.sh)
- [`counted_vs_pinned.py`](counted_vs_pinned.py)

Result files the index names:

- `counted_vs_pinned_q1..q4_2026-10-15.json` (not in this folder at the source commit)
- `box4_s4r_manifest_provenance_v2_2026-10-15.json` (not copied: result left out to keep the release under 120 MB)

**As-run check.** This folder's committed result files record no code commit, so we could not check from them whether the copies here are the versions that ran.

## Files here

84 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-15_q3_simhonest_311/` at commit `7935e9e0be05170e8946112a1e21fa039d2db951` of our internal repository. 212 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
