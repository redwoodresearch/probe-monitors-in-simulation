# 2026-10-12_claimB_c2_fresh_tasks

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

How many fresh task pairs (never-used LinuxArena main tasks paired with existing side tasks) could be built to confirm the environment-halves result (train on one half of the environments and test on the other, in both directions) on new data?

## What we did

A \$0 counting script (task IDs and counts only, no attack text) for never-used main tasks and admissible (main task, side task) pairs per environment and halves direction. Two attempts were stopped by hand when the used-task scan ran over 10 minutes; no result file was written.

## What we found

No result: commit messages 0e3f9bbb8 and 36bd8897f record runs stopped before writing any result; no later run of this count in the experiment log or spend ledger.

## In the post?

Not in the post.

## Status

Void (planning stage; never produced a count)

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/); [2026-10-11_claimB_halves_confirm](../2026-10-11_claimB_halves_confirm/)

## Main scripts

- [`count_c2_cells.py`](count_c2_cells.py)

## Files here

1 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_claimB_c2_fresh_tasks/` at commit `36bd8897fdfb52e86744bf57a493459ded8c9e51` of our internal repository. Every file in that folder was copied.

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
