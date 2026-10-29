# 2026-10-11_q2_reference_audit

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

Which real run was each simulated run built from, so that no real-versus-simulated result compares a simulated run with its own source?

## What we did

\$0 check of manifests, launch records and honest sample IDs: Sol simulated honest references against the real honest cohort (by sample ID and file hash), the source arm of each Qwen Phase 2 simulated run, and which arms the threshold-in-simulation inputs hold; post hoc completeness check.

## What we found

Commits 277575843 and f5abab59f: 0 of 437 Sol simulated-honest references are real-cohort runs; all Qwen simulated-attack references are runs of the first real honest arm (212 after the completeness check, acb562c50); the threshold-in-simulation rows hold only real attack and simulated attack arms.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-06_astra_sim_honest](../2026-10-06_astra_sim_honest/); [2026-10-08_q3_onpolicy_figures](../2026-10-08_q3_onpolicy_figures/); [2026-10-09_q2_threshold_in_sim](../2026-10-09_q2_threshold_in_sim/)

## Main scripts

- [`check_references.py`](check_references.py)
- [`check_arms.py`](check_arms.py)
- [`check_complete_POSTHOC.py`](check_complete_POSTHOC.py)

Result files the index names:

- `reference_audit_2026-10-11_194008.json` (not copied: result left out to keep the release under 120 MB)
- [`arms_2026-10-11_194523.json`](results/arms_2026-10-11_194523.json)
- `complete_POSTHOC_2026-10-12_042237.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-11_q2_reference_audit/` at commit `2c7faeeeff83f12356be5ef2d9e3973f21ed2d05` of our internal repository. 2 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
