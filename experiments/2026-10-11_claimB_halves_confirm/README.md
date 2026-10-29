# 2026-10-11_claimB_halves_confirm

2026-10-11 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Can the environment-halves strict-alarm result (a realism probe trained on half of the environments and tested on the other half flagged more formatting-fixed simulated runs than second real runs at a 1% false-alarm threshold) be confirmed on fresh data, and does run length explain it?

## What we did

\$0 planning: checked whether runs already bought (50 new GPT-6-Astra simulated honest runs from a 230-run purchase, and new real GPT-5.6-Sol honest runs from the credential-monitor purchase) could confirm it, priced a comparable fresh design, and ran an exploratory length-control analysis (length strata and length-only baselines) on the stored halves scores.

## What we found

NOTE.md: the existing fresh runs cannot confirm it (honest mode, unfixed September simulator, attack-calibrated threshold); a comparable design needs about 150-200 cells per direction, about \$5.0-6.6k with 30% buffer. Commit aa2521b9d: the gap persists after coarse length adjustment (pooled 3.8 points [2.3, 5.4]); length alone AUROC 0.54 and 0.56.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-11_claimB_halves](../2026-10-11_claimB_halves/)

## Main scripts

- [`analyze_power_confirm.py`](analyze_power_confirm.py)
- [`analyze_length_control.py`](analyze_length_control.py)
- [`count_attack_material.py`](count_attack_material.py)

Result files the index names:

- `power_confirm_2026-10-11_234202.json` (not copied: result left out to keep the release under 120 MB)
- `length_control_2026-10-11_235708.json` (not copied: result left out to keep the release under 120 MB)
- `attack_material_2026-10-11_235933.json` (not copied: result left out to keep the release under 120 MB)
- `codex_factcheck_length_control_2026-10-12.txt` (not copied: notes, reviews, prompts, examples, labels, judge outputs or logs (by file name))

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_claimB_halves_confirm/` at commit `18761497e1826f7ad56ad403b1dccf8326c016b1` of our internal repository. 7 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
