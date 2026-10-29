# 2026-10-13_q2_item3_probe_sees

2026-10-13 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Is the attack instruction or side-task text visible to the activation readers in attack transcripts?

## What we did

Code reading of every reader's staging (the attack system prompt is replaced by the honest one; no tools block), then scripted counts on Modal CPU over every staged input folder: own side-task document windows, any environment side task, attacker-prompt lines and generic markers, per message part, with honest arms as controls, rendered per reader. About \$0.03.

## What we found

Codex review of v1 (codex_reviews_item3_2026-10-13.md): normalized system message equals the honest prompt for 2,933 of 2,933 retained attack runs, and zero system or user hits for the run's own side task, any same-environment side task, or attacker-prompt lines; one attack run's tool outputs matched a fragment of its own side-task document. Later versions (v2 to v5) extended scope; their JSONs are on S3.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

Staged inputs of [2026-10-06_segment_end_extraction](../2026-10-06_segment_end_extraction/) and later stagings

## Main scripts

- [`run_probe_sees.py`](run_probe_sees.py)
- [`run_probe_sees_v2.py`](run_probe_sees_v2.py)
- [`run_probe_sees_v3.py`](run_probe_sees_v3.py)
- [`run_probe_sees_v4.py`](run_probe_sees_v4.py)
- [`run_probe_sees_v5.py`](run_probe_sees_v5.py)
- [`make_assets_manifest.py`](make_assets_manifest.py)

Result files the index names:

- `own_words_keys_v5_POSTHOC_2026-10-13_232137.json` (not copied: result left out to keep the release under 120 MB)
- `assets_manifest_POSTHOC_2026-10-14_110613.json` (not copied: result left out to keep the release under 120 MB)
- `s3_manifest_v1.json` (not copied: strings over 300 characters)

## Files here

6 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q2_item3_probe_sees/` at commit `efc615f383c7712112e4122d63ad8c4870414814` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
