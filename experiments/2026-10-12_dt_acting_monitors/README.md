# 2026-10-12_dt_acting_monitors

2026-10-12 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Do destructive-command and data-transfer monitors on Qwen3.5-27B's own 196-task acting runs catch as often in simulated runs as in real runs, beside the gap between two real runs?

## What we did

Built a blind label sheet of 74,873 calls, then fit command-level probes for destructive and transfer labels (exact pattern labels, then provisional AI-judge labels) on H100s on Modal, comparing first real, simulated and second real attack runs at 1% and 5% alarms with planted 5-point shifts and label-shuffle nulls.

## What we found

Exact labels give few positives; destructive primary real minus simulated at 1% -14.8 points [-26.1, +0.5] with the planted shift not detected; transfer +0.0 [-1.1, +1.8]; on provisional AI-judge labels every range includes 0 (e.g. destructive secondary at 5% +0.1 [-0.2, +0.3]).

## In the post?

Not in the post.

## Status

Exploratory (AI-judge labels provisional)

## Builds on

[2026-10-12_destructive_transfer_labels](../2026-10-12_destructive_transfer_labels/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/) (196-task acting runs)

## Main scripts

- [`run_dt_attack_fits.py`](run_dt_attack_fits.py)
- [`analyze_dt_attack_fits.py`](analyze_dt_attack_fits.py)
- [`make_label_sheet.py`](make_label_sheet.py)
- [`make_blind_sheet_v3.py`](make_blind_sheet_v3.py)
- [`count_label_positives.py`](count_label_positives.py)
- [`note_dt_attack_fits.py`](note_dt_attack_fits.py)

Result files the index names:

- `RESULTS_dt_attack_fits.md` (not copied: notes or README (internal working labels))
- `manifest_dt_attack_fits_v1.json` (not copied: strings over 300 characters)
- `manifest_dt_attack_semantic_fits_v1.json` (not copied: strings over 300 characters)

## Files here

14 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-12_dt_acting_monitors/` at commit `fdddb15dbdb4e90f6171099b6766fe3c55c8ba66` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
