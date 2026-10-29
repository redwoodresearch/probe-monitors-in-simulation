# 2026-10-12_claimB_meaning_judge

2026-10-12 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Does the unrepaired simulator report the same facts as a real run as often as a second real run does, judged by a stronger AI judge on honest runs?

## What we did

Found same-state sites where two real honest runs issue the same command after identical history; GPT-5.6-Sol (Gemini 3.1 Pro attempted) judged whether the simulated reply and the second real reply report the same facts as the first real reply, in both orders, with planted meaning-breaking controls.

## What we found

RESULT.md: planted check passes (0.955 judged different); 33 primary sites: agreement difference -0.061 (90% interval [-0.129, +0.000]); 58 secondary sites -0.069 [-0.135, -0.018]; verdict "too few"; Gemini closed after rate-limit errors.

## In the post?

Not in the post.

## Status

Exploratory (preliminary); two interim analysis files superseded inside the folder

## Builds on

[2026-10-06_matched_step_pairs](../2026-10-06_matched_step_pairs/); [2026-10-11_claimB_k_replies](../2026-10-11_claimB_k_replies/)

## Main scripts

- [`run_meaning_judge.py`](run_meaning_judge.py)
- [`run_sim_sites.py`](run_sim_sites.py)
- [`analyze_meaning.py`](analyze_meaning.py)
- [`analyze_prefix_matched.py`](analyze_prefix_matched.py)
- [`build_sites.py`](build_sites.py)
- [`make_plants.py`](make_plants.py)

Result files the index names:

- `meaning_result_2026-10-12_164829.json` (not in this folder at the source commit)
- `prefix_matched_2026-10-12_145915.json` (not copied: result left out to keep the release under 120 MB)
- [`plants_2026-10-12_150409.json`](results/plants_2026-10-12_150409.json)
- `RESULT.md` (not copied: notes or README (internal working labels))

## Files here

10 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-12_claimB_meaning_judge/` at commit `bd2f7e909c13f71f6da160b56a584316ccd690ad` of our internal repository. 18 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
