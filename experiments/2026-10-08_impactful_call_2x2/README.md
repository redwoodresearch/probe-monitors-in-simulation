# 2026-10-08_impactful_call_2x2

2026-10-08 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Does the impactful-command probe catch impactful commands as well on simulated runs as on real ones when Qwen3.5-27B (or Qwen3-32B) reads GPT-5.6-Sol's transcripts (the off-policy 2x2)?

## What we did

From saved scores (\$0): impactful vs read-only calls in attack runs, probes trained on real or simulated runs and tested on both, read at the last content token of each command, reporting catch at 1% false alarms, AUROC and standardized partial AUROC (pAUC), with a content-token null; made meeting figures for the project manager.

## What we found

For Qwen3.5-27B catch at 1% is 88.9% real vs 87.7% simulated (gap -1.2 points [-10.3, +7.4]), AUROC 0.991 / 0.992; Qwen3-32B's gap is -3.5 points (preliminary).

## In the post?

Not in the post.

## Status

Current (preliminary; refit version in [2026-10-08_impactful_refit_bootstrap](../2026-10-08_impactful_refit_bootstrap/))

## Builds on

[2026-10-06_q2_impactful_call_transfer](../2026-10-06_q2_impactful_call_transfer/); [2026-10-06_qwen35_replication](../2026-10-06_qwen35_replication/)

## Main scripts

- [`analyze_pauc_2x2.py`](analyze_pauc_2x2.py)
- [`run_content_null.py`](run_content_null.py)
- [`contrast_table.py`](contrast_table.py)
- [`plot_2x2.py`](plot_2x2.py)
- [`plot_figure2.py`](plot_figure2.py)
- [`plot_oct7_2x2.py`](plot_oct7_2x2.py)

Result files the index names:

- `summary_2026-10-08.json` (not copied: strings over 300 characters)
- `manifest_pauc_v5.json` (not copied: result left out to keep the release under 120 MB)
- `impactful_2x2_james_2026-10-08_115003.png` (not copied: result file type not copied (.png))
- `oct7_2x2_with_intervals_2026-10-08_115432.png` (not copied: result file type not copied (.png))

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-08_impactful_call_2x2/` at commit `949249dbc0ad61fe6320d5d05db9b942888f26da` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
