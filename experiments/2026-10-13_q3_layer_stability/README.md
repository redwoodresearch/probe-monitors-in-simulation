# 2026-10-13_q3_layer_stability

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Is the simulation-only training recipe's layer choice unstable: does averaging the top two simulation-chosen layers shrink the downside of a wrong pick, and is there a stable range of good layers?

## What we did

(a) Averaged the scores of the top two layers chosen on simulated data and reran 20 reseeds against the count-matched real-trained detector; (b) ran the count-matched previously selected main probe configuration (full repair, averaged readout) at every extracted layer for Seed-OSS-36B (7 layers) and Qwen3-32B. GPT-5.6-Sol runs. Modal.

## What we found

layers_a_v2_2026-10-13_235012.json: at 1% the unit-level downside of a non-48 pick falls from 0.093 to 0.004 with two layers ('shrinks' true). layers_b: layers whose lower end of D (the catch difference against the count-matched real-trained probe) is above -10 at 1%: Seed 24, 32, 40, 48; Qwen3-32B 24, 48, 56; at 5% Seed 24 to 63 and Qwen 16 to 56; 'only 48' false for both.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (recipe 1, cells); [2026-10-13_q2_seed_callend](../2026-10-13_q2_seed_callend/) (Seed extraction check)

## Main scripts

- [`run_layers_a.py`](run_layers_a.py)
- [`run_layers_a_v2.py`](run_layers_a_v2.py)
- [`run_layers_a_like.py`](run_layers_a_like.py)
- [`run_layers_b.py`](run_layers_b.py)
- [`check_dg_sha.py`](check_dg_sha.py)

Result files the index names:

- `layers_a_2026-10-13_233549.json` (not copied: result left out to keep the release under 120 MB)
- `layers_a_v2_2026-10-13_235012.json` (not copied: result left out to keep the release under 120 MB)
- `layers_a_like_2026-10-14_002544.json` (not copied: result left out to keep the release under 120 MB)
- `layers_b_qwen_2026-10-13_233027.json` (not copied: result left out to keep the release under 120 MB)
- `layers_b_seed_2026-10-13_233154.json` (not copied: result left out to keep the release under 120 MB)
- [`check_dg_sha_2026-10-14_004435.json`](results/check_dg_sha_2026-10-14_004435.json)

## Files here

7 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_q3_layer_stability/` at commit `d99aebd7f33e9c46ec3c10df65f16345667ac986` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
