# 2026-10-11_credential_count

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

The credential-monitor study: how many tool outputs contain credentials in real and simulated GPT-5.6-Sol runs, and does the committed positive-count gate give enough power to compare a credential monitor between them?

## What we did

Preserves the regex credential detector and extraction runner that produced the counts in a pull request (run locally outside Claude), and a regrouping script that applies the corrected 163-task same-batch real-versus-real control to frozen numeric labels; no API or GPU job.

## What we found

GATE.md: simulated 1,385 positives over 437 runs, matched real 2,859 over 800, same-batch real runs 546 and 578 over 163; predicted chance of agreement at most 0.700, below 0.80, so 'C is too few to tell'.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_label_artifacts](../2026-10-11_label_artifacts/); [2026-10-11_q2_zoo_plans](../2026-10-11_q2_zoo_plans/)

## Main scripts

- [`analyze_credential_count.py`](analyze_credential_count.py)
- [`analyze_credential_gate.py`](analyze_credential_gate.py)

Result files the index names:

- `gate_counts_v3.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_credential_count/` at commit `e37a2892f4b2a76dce4deacc2b4b4a1bbe9cd35f` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
