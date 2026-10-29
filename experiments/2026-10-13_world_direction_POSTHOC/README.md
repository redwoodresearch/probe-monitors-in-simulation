# 2026-10-13_world_direction_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

If the honest-run 'world' direction (what separates simulated from real honest runs) is projected out before training on simulated data, does the simulation-trained detector catch real attacks as well as a real-trained one?

## What we did

Per nested model, fit honest-only world directions (logistic nullspace k = 1 to 10, and the mean difference), removed them, and refit the fully fixed simulated/simulated and mixed recipes of the Qwen3-32B tool-output probe (GPT-5.6-Sol runs, layers 48 and 63), gated on k = 0 reproducing the stored fit. Modal H100, \$0.42.

## What we found

README: removing the world direction does not recover simulated/simulated real catch; at 1%, D (second-real-run catch under the tested probe minus catch under the full-count real-trained probe) ranges from -17.4 to -15.9 at layer 48 and -13.8 to -11.9 at layer 63 across all seven variants; the simulated-minus-real gap widens; realism left at k = 10 stays above its null (0.583 vs matched 95th 0.522 at layer 48).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (another working session's full-fix grid); [2026-10-13_realism_by_repair_POSTHOC](../2026-10-13_realism_by_repair_POSTHOC/) (null results)

## Main scripts

- [`run_world_direction.py`](run_world_direction.py)
- [`analyze_world_direction.py`](analyze_world_direction.py)

Result files the index names:

- [`worlddir_metrics_POSTHOC_2026-10-13_171503.printout.txt`](results/worlddir_metrics_POSTHOC_2026-10-13_171503.printout.txt)
- [`worlddir_metrics_L63_POSTHOC_2026-10-13_171813.printout.txt`](results/worlddir_metrics_L63_POSTHOC_2026-10-13_171813.printout.txt)
- `manifest_worlddir_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

5 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_world_direction_POSTHOC/` at commit `da99ac0d58e4546cfc1e1032aed25c38d4a82e38` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
