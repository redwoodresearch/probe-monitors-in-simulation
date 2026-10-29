# 2026-10-13_recipe_transfer_POSTHOC

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Does the simulation-trained recipe, trained off-policy on GPT-5.6-Sol's runs, transfer when tested on Qwen3.5-27B's own runs?

## What we did

Trained the recipe's probe variants (real-trained, simulated attack vs simulated honest, and mixed simulated attack vs real honest, with editor-output-repaired variants of the two simulation-trained recipes) on GPT-5.6-Sol's runs read by Qwen3.5-27B (last-token tool-output readout) and tested them on Qwen3.5-27B's own 183-task cohort with thresholds from its own honest runs, on full and reasoning-stripped transcripts. Modal.

## What we found

README (recipe_transfer_POSTHOC_2026-10-14_000532.json): on own full transcripts at 1%, the mixed recipe overstates by +27.9 [11.6, 35.9] points while simulated/simulated is +2.2 [-2.7, 8.2]; every second-real-run catch difference against the real-trained or count-matched detector is 'inconclusive' (intervals cross 0 and extend past +/-10); real catch is low (8 to 14%).

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (recipe); [2026-10-13_onpolicy_reply](../2026-10-13_onpolicy_reply/) (own-run caches)

## Main scripts

- [`run_recipe_transfer.py`](run_recipe_transfer.py)
- [`make_readme.py`](make_readme.py)

Result files the index names:

- [`recipe_transfer_POSTHOC_2026-10-14_000532.json`](results/recipe_transfer_POSTHOC_2026-10-14_000532.json)
- `recipe_transfer_POSTHOC_2026-10-14_000532.printout.txt` (not copied: lines over 300 characters)
- `manifest_recipe_transfer_v1.json` (not copied: result left out to keep the release under 120 MB)

## Files here

4 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_recipe_transfer_POSTHOC/` at commit `73dbcb518afaf4ad3eb17f31dfa56625ac4b9df8` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
