# 2026-10-17_records_step_container

2026-10-17 · part of [Infrastructure and tooling](../../docs/experiments.md#infrastructure-and-tooling)

## Question

Run allowlisted record-builder scripts unchanged inside a Modal container so the laptop never parses value records.

## What we did

Records-step wrapper (git-bundle in, metadata out), its suffix variant for concurrent steps, and step specs for the pilot records and the final-pipeline catalogue steps.

## What we found

The pilot provenance-record and verification-record steps ran, settled \$0.000 each (spend rows).

## In the post?

Not in the post.

## Status

Current (launch plumbing)

## Builds on

[2026-10-15_cut_sensitivity](../2026-10-15_cut_sensitivity/); [2026-10-14_q3_overnight_chain](../2026-10-14_q3_overnight_chain/)

## Main scripts

- [`run_records_step_in_container.py`](run_records_step_in_container.py)
- [`run_records_step_in_container_sfx.py`](run_records_step_in_container_sfx.py)
- [`make_dryrun_sidecar.py`](make_dryrun_sidecar.py)

## Files here

7 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-17_records_step_container/` at commit `607a1d6083cecdfb6850f143d05248ddbfed96f0` of our internal repository. 30 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
